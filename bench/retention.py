"""Small paired, executable-answer screen; not a general-intelligence benchmark."""
import argparse
import asyncio
import itertools
import json
from pathlib import Path
import random
import time

import aiohttp
from transformers import AutoTokenizer

from concurrency import run_one, engine_counters


def cases():
    rows = []
    for seed in (1709, 2713, 3911):
        rng = random.Random(seed)
        # Exhaustive reference for a bounded scheduling problem.
        jobs = [(i, rng.randrange(0, 18), rng.randrange(2, 7), rng.randrange(2, 15))
                for i in range(8)]
        best = 0
        for mask in range(1 << len(jobs)):
            selected = sorted((s, s+d, v) for i,s,d,v in jobs if mask & (1 << i))
            if all(a[1] <= b[0] for a,b in zip(selected, selected[1:])):
                best = max(best, sum(j[2] for j in selected))
        rows.append((f'schedule-{seed}',
            'One machine runs a subset of fixed-time jobs. Jobs are [id, start, duration, value]. '
            'Intervals are half-open, jobs cannot overlap or move, and each can run once. '
            f'Jobs: {json.dumps(jobs)}. What is the maximum total value? Return {{"answer": integer}}.', best))
        # All satisfying assignments, rather than an LLM-produced oracle.
        a,b = rng.sample(range(1,7),2)
        valid = [p for p in itertools.permutations(range(1,7))
                 if p.index(a) < p.index(b) and p[0] % 2 == 0
                 and p[1] + p[4] == 7 and abs(p.index(1)-p.index(6)) > 1]
        rows.append((f'logic-{seed}',
            'Permute the integers 1 through 6, each exactly once. Positions are 1-indexed. '
            f'{a} appears before {b}; the first integer is even; integers in positions 2 and 5 '
            'sum to 7; integers 1 and 6 are not adjacent. How many permutations satisfy all '
            'constraints? Return {"answer": integer}.', len(valid)))
        grid = [[rng.randrange(1,10) for _ in range(5)] for _ in range(5)]
        dp = [[0]*5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                dp[i][j] = grid[i][j] + (min(dp[i-1][j] if i else 10**9,
                                            dp[i][j-1] if j else 10**9) if i or j else 0)
        rows.append((f'path-{seed}',
            f'Grid costs: {json.dumps(grid)}. Move from top-left to bottom-right, only right '
            'or down. Include both endpoints in the total. What is the minimum total cost? '
            'Return {"answer": integer}.', dp[-1][-1]))
        # Python aliasing and control flow with a directly executable oracle.
        values = [rng.randrange(2,10) for _ in range(4)]
        shared = values[:]
        result = [shared, shared[:], shared]
        result[0].append(result[1].pop())
        result[2][1] += result[1][0]
        expected = [sum(x) for x in result]
        code = (f'a = {values!r}\nb = [a, a[:], a]\n'
                'b[0].append(b[1].pop())\nb[2][1] += b[1][0]\n'
                'print([sum(x) for x in b])')
        rows.append((f'python-{seed}',
            f'What does this Python 3 program print?\n```python\n{code}\n```\n'
            'Return {"answer": [integer, integer, integer]}.', expected))
    return rows


def parse_answer(text):
    answer = text.split('</think>')[-1].strip()
    if answer.startswith('```') and answer.endswith('```'):
        answer = '\n'.join(answer.splitlines()[1:-1]).strip()
    try:
        return json.loads(answer)['answer']
    except (ValueError, KeyError, TypeError):
        return None


async def main(a):
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    rows = cases()
    semaphore = asyncio.Semaphore(a.concurrency)
    barrier = asyncio.Event()
    barrier.set()
    a.mode = 'retention'
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=a.timeout)) as session:
        before = await engine_counters(session,a.url)
        async def one(i,case):
            name,prompt,expected = case
            ids = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                                tokenize=True,add_generation_prompt=True)
            async with semaphore:
                row = await run_one(session,a.url,i,ids,expected,a,barrier)
            row.update(name=name,prompt=prompt,parsed_answer=parse_answer(row['text']))
            row['correct'] = (not row['error'] and row['finish_reason']=='stop'
                              and row['parsed_answer']==expected)
            return row
        started = time.monotonic()
        results = await asyncio.gather(*(one(i,c) for i,c in enumerate(rows)))
        wall = time.monotonic()-started
        after = await engine_counters(session,a.url)
    report = {'configuration':a.label,'model':a.model,'cases':results,
              'summary':{'correct':sum(r['correct'] for r in results),'total':len(results),
                         'errors':sum(bool(r['error']) for r in results),
                         'all_attempt_output_tokens':sum((r['usage'] or {}).get('completion_tokens',0) for r in results),
                         'wall_seconds':wall,'scope':'small procedural retention screen only'},
              'engine_counters_before':before,'engine_counters_after':after}
    if a.reference:
        reference = json.loads(Path(a.reference).read_text())
        old = {r['name']:r for r in reference['cases']}
        if set(old) != {r['name'] for r in results} or any(old[r['name']]['prompt'] != r['prompt'] for r in results):
            raise RuntimeError('Reference prompts differ; refusing an unpaired comparison')
        report['paired'] = {
            'reference':str(a.reference),
            'regressions':[r['name'] for r in results if old[r['name']]['correct'] and not r['correct']],
            'gains':[r['name'] for r in results if not old[r['name']]['correct'] and r['correct']],
            'identical_output_tokens':sum(old[r['name']]['output_token_ids']==r['output_token_ids'] for r in results)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with Path(a.output).open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report['summary'],indent=2))
    return int(report['summary']['errors'] > 0)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--output',required=True)
    p.add_argument('--reference')
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--output-tokens',type=int,default=4096)
    p.add_argument('--timeout',type=int,default=1200)
    args=p.parse_args()
    if min(args.concurrency,args.output_tokens,args.timeout)<=0:
        p.error('concurrency, token limit, and timeout must be positive')
    if Path(args.output).exists():
        p.error('output already exists; preserve previous measurements')
    raise SystemExit(asyncio.run(main(args)))
