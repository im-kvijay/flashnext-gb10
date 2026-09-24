"""Everyday-assistant checks through the chat API, each graded automatically.

Covers what an agent harness relies on beyond benchmarks: arithmetic, exact
format instructions, JSON output, multi-turn memory, runnable code (executed
against tests), a tool call followed by use of the tool result, summarizing a
given text without losing its key fact, and translation. The server's default
sampling (thinking enabled) is used, as a user would; each task runs
--repeats times and reports passes, so a damaged configuration shows up as
failures rather than as a small score shift.
"""
import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile

import aiohttp

TOOLS = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Current weather for a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
ARTICLE = ('The Lindqvist Bridge, completed in 1931, spans 412 meters across the Varn estuary. Engineers chose '
           'a cantilever design because the riverbed could not support intermediate piers. In 2019 the city '
           'closed it to trucks after inspectors found corrosion in two of its eight main anchor cables. '
           'Repairs are scheduled to finish in 2027, and until then only cars and bicycles may cross.')


def final_text(msg):
    return (msg.get('content') or '').strip()


def run_python(code, tests):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code + '\n\n' + tests + '\n')
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def code_block(text):
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    return blocks[-1] if blocks else text


async def chat(session, url, messages, tools=None, max_tokens=8192):
    body = {'model': 'flashnext', 'messages': messages, 'max_tokens': max_tokens}
    if tools:
        body['tools'] = tools
    async with session.post(url + '/v1/chat/completions', json=body) as r:
        r.raise_for_status()
        return (await r.json())['choices'][0]['message']


async def t_arithmetic(s, url):
    m = await chat(s, url, [{'role': 'user', 'content': 'A shop sells pens at $3 each and notebooks at $7 each. '
                                                         'Maya buys 4 pens and 3 notebooks and pays with a $50 bill. '
                                                         'How much change does she get? Answer with just the number.'}])
    return re.sub(r'[^\d.]', '', final_text(m)).rstrip('.') in ('17', '17.00', '17.0'), final_text(m)


async def t_format(s, url):
    m = await chat(s, url, [{'role': 'user', 'content': 'List exactly three benefits of unit tests as a markdown '
                                                         'bulleted list using "- ". No other text before or after.'}])
    lines = [l for l in final_text(m).splitlines() if l.strip()]
    return len(lines) == 3 and all(l.lstrip().startswith('- ') for l in lines), final_text(m)


async def t_json(s, url):
    m = await chat(s, url, [{'role': 'user', 'content': 'Extract the people from this sentence as JSON: a list of '
                                                         'objects with keys "name" (string) and "age" (integer). '
                                                         'Output only the JSON.\n\nSentence: Priya, who is 34, '
                                                         'hired Tomas (27) and later his sister Elena, aged 41.'}])
    text = final_text(m)
    try:
        data = json.loads(re.sub(r'^```(?:json)?\n|\n?```$', '', text.strip()))
        got = sorted((d['name'], d['age']) for d in data)
        return got == [('Elena', 41), ('Priya', 34), ('Tomas', 27)], text
    except Exception:
        return False, text


async def t_memory(s, url):
    msgs = [{'role': 'user', 'content': 'For this conversation: our deployment codename is BLUE-HERON-7 and the '
                                        'release date is March 14. Just acknowledge briefly.'}]
    m = await chat(s, url, msgs)
    msgs += [{'role': 'assistant', 'content': final_text(m)},
             {'role': 'user', 'content': 'Unrelated: what is the capital of Australia? One word.'}]
    m2 = await chat(s, url, msgs)
    msgs += [{'role': 'assistant', 'content': final_text(m2)},
             {'role': 'user', 'content': 'What was the deployment codename and the release date I gave you?'}]
    m3 = await chat(s, url, msgs)
    t = final_text(m3)
    return 'Canberra' in final_text(m2) and 'BLUE-HERON-7' in t and 'March 14' in t, t


async def t_code(s, url):
    m = await chat(s, url, [{'role': 'user', 'content': 'Write a Python function merge_intervals(intervals) that '
                                                         'takes a list of [start, end] pairs and returns the merged, '
                                                         'sorted list of overlapping intervals (touching intervals '
                                                         'like [1,2] and [2,3] merge). Reply with one python code '
                                                         'block only.'}], max_tokens=12288)
    tests = ('assert merge_intervals([]) == []\n'
             'assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n'
             'assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n'
             'assert merge_intervals([[5,6],[1,2],[2,3]]) == [[1,3],[5,6]]\n'
             'assert merge_intervals([[1,10],[2,3]]) == [[1,10]]\n')
    code = code_block(final_text(m))
    return run_python(code, tests), code[-300:]


async def t_tool(s, url):
    msgs = [{'role': 'user', 'content': 'Should I bring an umbrella in Oslo today? Check the weather first.'}]
    m = await chat(s, url, msgs, TOOLS)
    calls = m.get('tool_calls') or []
    if not calls or calls[0]['function']['name'] != 'get_weather':
        return False, f'no tool call: {final_text(m)[:200]}'
    args = json.loads(calls[0]['function']['arguments'])
    msgs += [{'role': 'assistant', 'content': m.get('content') or '', 'tool_calls': calls},
             {'role': 'tool', 'tool_call_id': calls[0]['id'],
              'content': json.dumps({'city': args.get('city'), 'condition': 'heavy rain', 'precip_mm': 14,
                                     'temp_c': 9})}]
    m2 = await chat(s, url, msgs, TOOLS)
    t = final_text(m2)
    return 'oslo' in str(args.get('city', '')).lower() and bool(re.search(r'\byes\b|umbrella|rain', t, re.I)), t


async def t_summary(s, url):
    m = await chat(s, url, [{'role': 'user', 'content': f'Summarize in two sentences for a truck driver:\n\n{ARTICLE}'}])
    t = final_text(m)
    return bool(re.search(r'truck', t, re.I)) and bool(re.search(r'2027', t)) and len(t) < 700, t


async def t_translate(s, url):
    m = await chat(s, url, [{'role': 'user', 'content': 'Translate into French, output only the translation: '
                                                         '"Good morning, the meeting is postponed until tomorrow."'}])
    t = final_text(m).lower()
    return 'bonjour' in t and 'demain' in t and ('réunion' in t or 'reunion' in t), t


TASKS = [t_arithmetic, t_format, t_json, t_memory, t_code, t_tool, t_summary, t_translate]


async def main(a):
    results = {t.__name__[2:]: [] for t in TASKS}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        sem = asyncio.Semaphore(a.concurrency)

        async def one(task):
            async with sem:
                try:
                    ok, detail = await task(s, a.url)
                except Exception as e:  # a server error is a failure, not a crash
                    ok, detail = False, f'error: {e}'
                results[task.__name__[2:]].append({'ok': bool(ok), 'detail': str(detail)[-400:]})
        await asyncio.gather(*(one(t) for t in TASKS for _ in range(a.repeats)))
    summary = {k: f"{sum(r['ok'] for r in v)}/{len(v)}" for k, v in results.items()}
    passed = sum(r['ok'] for v in results.values() for r in v)
    total = sum(len(v) for v in results.values())
    report = {'summary': summary, 'passed': passed, 'total': total, 'results': results}
    if a.output:
        open(a.output, 'w').write(json.dumps(report, indent=2))
    print(json.dumps({'summary': summary, 'passed': f'{passed}/{total}'}, indent=2))
    for k, v in results.items():
        for r in v:
            if not r['ok']:
                print(f'FAIL {k}: {r["detail"][-200:]!r}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--model', help='unused; accepted for harness compatibility')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--output')
    asyncio.run(main(p.parse_args()))
