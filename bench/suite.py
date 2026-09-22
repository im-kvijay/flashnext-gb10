"""Task-level quality screen for serving configurations, weighted toward hard coding.

LiveCodeBench release 6 (January-April 2025 contest problems; mostly hard,
graded on all public and hidden tests) is the primary signal. HumanEval, GSM8K
and MMLU-Pro are small sanity checks. Uses the served chat template with
thinking enabled, greedy decoding, and 16 requests in flight. HumanEval is
graded by executing its tests; GSM8K by the final number; MMLU-Pro by the
answer letter. Responses that hit the token limit
count as wrong for every configuration alike. --compare reports paired flips
between two runs, which is what matters for detecting a regression: with a
few hundred items, aggregate accuracy differences under about 3 points are
within noise, while a precision change that damages the model shows up as
many one-directional flips. Use bench/fidelity.py for small numerical effects.

Data (scripts/prepare_eval_data.py): LiveCodeBench code_generation_lite
test6.jsonl (default 10 AtCoder hard, 10 LeetCode hard, 5 each medium, in file
order), openai_humaneval (first 40), gsm8k main (fixed-seed sample, first 20)
and MMLU-Pro (fixed-seed sample, 2 per category).
"""
import argparse
import asyncio
import base64
import json
import pickle
import zlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp

LETTERS = 'ABCDEFGHIJKLMNOP'


LCB_STDIN = ('You will be given a question (problem specification) and will generate a correct Python '
             'program that matches the specification and passes all tests.\n\nQuestion: {question}\n\n'
             'Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly '
             'test on the sample inputs). Enclose your code within delimiters as follows.\n'
             '```python\n# YOUR CODE HERE\n```')
LCB_FUNCTION = ('You will be given a question (problem specification) and will generate a correct Python '
                'program that matches the specification and passes all tests.\n\nQuestion: {question}\n\n'
                'You will use the following starter code to write the solution to the problem and enclose '
                'your code within delimiters.\n```python\n{starter}\n```')
LCB_PRELUDE = ('import sys, math, heapq, bisect, itertools, functools, collections, string, re\n'
               'from typing import *\nfrom collections import *\nfrom itertools import *\n'
               'from functools import *\nfrom heapq import *\nfrom bisect import *\nfrom math import *\n'
               'sys.setrecursionlimit(1 << 25)\n')


def lcb_items(path, quotas):
    taken = {}
    for row in map(json.loads, open(path)):
        key = (row['platform'], row['difficulty'])
        if taken.get(key, 0) >= quotas.get(key, 0):
            continue
        taken[key] = taken.get(key, 0) + 1
        tests = json.loads(row['public_test_cases'])
        tests += json.loads(pickle.loads(zlib.decompress(base64.b64decode(row['private_test_cases']))))
        function = json.loads(row['metadata'] or '{}').get('func_name') if isinstance(row['metadata'], str) \
            else (row['metadata'] or {}).get('func_name')
        template = LCB_FUNCTION if row['starter_code'] else LCB_STDIN
        yield dict(id=f"lcb-{row['question_id']}", task='livecodebench',
                   prompt=template.format(question=row['question_content'], starter=row['starter_code']),
                   data=dict(tests=tests, function=function, difficulty=row['difficulty'],
                             platform=row['platform']))


LCB_FUNCTION_RUNNER = """
import json, sys
cases = json.load(open(sys.argv[1]))
solver = Solution()
method = getattr(solver, sys.argv[2])
for index, (arguments, expected) in enumerate(cases):
    got = method(*[json.loads(line) for line in arguments.split('\\n')])
    expected = json.loads(expected)
    if isinstance(got, tuple):
        got = list(got)
    same = got == expected or (isinstance(got, float) and isinstance(expected, (int, float))
                               and abs(got - expected) <= 1e-6 * max(1, abs(expected)))
    if not same:
        print(f'case {index} failed', file=sys.stderr)
        sys.exit(1)
"""


def grade_livecodebench(content, data):
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', content, re.S)
    if not blocks:
        return False, 'no code block'
    code = LCB_PRELUDE + blocks[-1]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'solution.py'
        if data['function']:
            cases = [[t['input'], t['output']] for t in data['tests']]
            (Path(tmp) / 'cases.json').write_text(json.dumps(cases))
            path.write_text(code + '\n' + LCB_FUNCTION_RUNNER)
            try:
                result = subprocess.run(['python3', '-I', str(path), 'cases.json', data['function']],
                                        cwd=tmp, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                return False, 'timeout'
            return result.returncode == 0, result.stderr.decode(errors='replace')[-200:]
        path.write_text(code)
        for index, test in enumerate(data['tests']):
            try:
                result = subprocess.run(['python3', '-I', str(path)], cwd=tmp, input=test['input'].encode(),
                                        capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                return False, f'case {index} timeout'
            got = result.stdout.decode(errors='replace').split()
            if result.returncode != 0 or got != test['output'].split():
                return False, f'case {index} failed'
    return True, f"{len(data['tests'])} tests"


def humaneval_items(path):
    for row in map(json.loads, open(path)):
        prompt = ('Complete the following Python function. Reply with the complete function, '
                  'including any imports it needs, in a single ```python code block.\n\n' + row['prompt'])
        yield dict(id=row['task_id'], task='humaneval', prompt=prompt, data=row)


def gsm8k_items(path):
    for i, row in enumerate(map(json.loads, open(path))):
        prompt = row['question'] + '\n\nSolve the problem. End your reply with a line "Final answer: <number>".'
        yield dict(id=f'gsm8k-{i}', task='gsm8k', prompt=prompt,
                   data={'answer': row['answer'].split('####')[-1].strip().replace(',', '')})


def mmlu_items(path):
    for row in map(json.loads, open(path)):
        options = '\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(row['options']))
        prompt = (f"{row['question']}\n\n{options}\n\n"
                  'Choose the single best option. End your reply with a line "Answer: <letter>".')
        yield dict(id=f"mmlu-{row['question_id']}", task='mmlu_pro', prompt=prompt,
                   data={'answer': row['answer'], 'category': row['category']})


def grade_humaneval(content, data):
    blocks = re.findall(r'```(?:python|py)?\n(.*?)```', content, re.S)
    if not blocks:
        return False, 'no code block'
    code = blocks[-1]
    imports = '\n'.join(l for l in data['prompt'].splitlines() if l.startswith(('import ', 'from ')))
    program = f"{imports}\n{code}\n\n{data['test']}\n\ncheck({data['entry_point']})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'candidate.py'
        path.write_text(program)
        try:
            result = subprocess.run(['python3', '-I', str(path)], cwd=tmp, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False, 'timeout'
    return result.returncode == 0, result.stderr.decode(errors='replace')[-300:]


def last_number(text):
    match = re.findall(r'Final answer:\s*\$?\\?(?:boxed\{)?(-?[\d,]*\.?\d+)', text, re.I)
    if not match:
        match = re.findall(r'(-?[\d,]*\.?\d+)', text)
    return match[-1].replace(',', '').rstrip('.') if match else None


def grade(item, content):
    if item['task'] == 'livecodebench':
        return grade_livecodebench(content, item['data'])
    if item['task'] == 'humaneval':
        return grade_humaneval(content, item['data'])
    if item['task'] == 'gsm8k':
        got = last_number(content)
        try:
            return got is not None and abs(float(got) - float(item['data']['answer'])) < 1e-6, got
        except ValueError:
            return False, got
    found = re.findall(r'Answer:\s*\**\(?([A-P])\b', content)
    got = found[-1] if found else None
    return got == item['data']['answer'], got


async def ask(session, a, item, semaphore):
    async with semaphore:
        payload = {'model': 'flashnext', 'messages': [{'role': 'user', 'content': item['prompt']}],
                   'max_tokens': a.max_tokens, 'temperature': 0}
        start = time.monotonic()
        try:
            for attempt in range(3):
                try:
                    async with session.post(a.url + '/v1/chat/completions', json=payload) as response:
                        body = await response.json()
                        if response.status != 200:
                            raise RuntimeError(str(body)[:300])
                    break
                except aiohttp.ClientConnectionError:
                    # A stale keep-alive connection fails before the request reaches the
                    # server; retrying keeps transport errors out of the quality comparison.
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)
            choice = body['choices'][0]
            content = choice['message'].get('content') or ''
            finish = choice['finish_reason']
            correct, detail = (grade(item, content) if finish == 'stop' else (False, 'length'))
            row = dict(id=item['id'], task=item['task'], correct=bool(correct), detail=str(detail)[:300],
                       finish=finish, completion_tokens=body['usage']['completion_tokens'],
                       content=content[-2000:], error=None)
        except Exception as exc:
            row = dict(id=item['id'], task=item['task'], correct=False, detail=None, finish=None,
                       completion_tokens=0, content='', error=str(exc)[:300])
        row['seconds'] = time.monotonic() - start
        return row


async def run(a):
    data = Path(a.data)
    mmlu, per_category = [], {}
    for item in mmlu_items(data / 'mmlu_pro.jsonl'):
        category = item['data']['category']
        if per_category.get(category, 0) < a.mmlu_per_category:
            per_category[category] = per_category.get(category, 0) + 1
            mmlu.append(item)
    quotas = {('atcoder', 'hard'): a.lcb_hard, ('leetcode', 'hard'): a.lcb_hard,
              ('atcoder', 'medium'): a.lcb_medium, ('leetcode', 'medium'): a.lcb_medium}
    items = (list(lcb_items(data / 'livecodebench_test6.jsonl', quotas))
             + list(humaneval_items(data / 'humaneval.jsonl'))[:a.humaneval]
             + list(gsm8k_items(data / 'gsm8k.jsonl'))[:a.gsm8k] + mmlu)
    if a.limit:
        items = items[:a.limit]
    semaphore = asyncio.Semaphore(a.concurrency)
    started = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=a.timeout)) as session:
        rows = await asyncio.gather(*(ask(session, a, item, semaphore) for item in items))
    summary = {'wall_seconds': time.monotonic() - started}
    for task in ('livecodebench', 'humaneval', 'gsm8k', 'mmlu_pro'):
        sel = [r for r in rows if r['task'] == task]
        if sel:
            summary[task] = dict(n=len(sel), accuracy=sum(r['correct'] for r in sel) / len(sel),
                                 length_limited=sum(r['finish'] == 'length' for r in sel),
                                 errors=sum(r['error'] is not None for r in sel),
                                 mean_completion_tokens=sum(r['completion_tokens'] for r in sel) / len(sel))
    Path(a.output).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + '\n')
    print(json.dumps(summary, indent=2))
    return 1 if any(r['error'] for r in rows) else 0


def compare(reference, candidate):
    ref = {r['id']: r for r in json.loads(Path(reference).read_text())['rows']}
    cand = {r['id']: r for r in json.loads(Path(candidate).read_text())['rows']}
    report = {}
    for task in ('livecodebench', 'humaneval', 'gsm8k', 'mmlu_pro', 'all'):
        ids = [i for i in ref if i in cand and (task == 'all' or ref[i]['task'] == task)]
        if not ids:
            continue
        lost = [i for i in ids if ref[i]['correct'] and not cand[i]['correct']]
        gained = [i for i in ids if cand[i]['correct'] and not ref[i]['correct']]
        report[task] = dict(n=len(ids), reference=sum(ref[i]['correct'] for i in ids) / len(ids),
                            candidate=sum(cand[i]['correct'] for i in ids) / len(ids),
                            lost=len(lost), gained=len(gained), lost_ids=lost[:20],
                            reference_tokens=sum(ref[i]['completion_tokens'] for i in ids),
                            candidate_tokens=sum(cand[i]['completion_tokens'] for i in ids))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--model', help='accepted for controller compatibility; unused')
    p.add_argument('--data', default=os.environ.get('FLASHNEXT_EVAL_DATA', 'data/eval'))
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--lcb-hard', type=int, default=10, help='per platform')
    p.add_argument('--lcb-medium', type=int, default=5, help='per platform')
    p.add_argument('--humaneval', type=int, default=40)
    p.add_argument('--gsm8k', type=int, default=20)
    p.add_argument('--mmlu-per-category', type=int, default=2)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--limit', type=int)
    p.add_argument('--timeout', type=int, default=14400)
    p.add_argument('--output')
    p.add_argument('--compare', nargs=2, metavar=('REFERENCE', 'CANDIDATE'))
    a = p.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=2))
    else:
        raise SystemExit(asyncio.run(run(a)))
