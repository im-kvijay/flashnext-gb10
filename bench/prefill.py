"""Prefill cost on real source code: cold fills and file-sized appends at depth.

Coding agents spend prefill in two ways: filling a fresh context (a new task
reading a repository) and appending tool output (a file read, a test log) to a
long, already-cached context. This measures both with distinct real source
text and server-verified token counts:

  cold-<n>     one uncached prompt of n tokens; time to first token
  append-<d>-<n>   a cached d-token context plus n new tokens (one agent)
  append8-<d>-<n>  eight agents append n new tokens each to the same cached
               d-token context at once (each suffix is distinct and attends
               to the full context; the shared prefix only saves priming time)

With --profile-dir set on the server, --profile brackets one window of the
cold 200k fill and the eight-way append with /start_profile and /stop_profile.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import aiohttp
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrency import CODEBASE_AGENTS, codebase_ids  # noqa: E402


async def complete(session, url, prompt):
    start = time.monotonic()
    payload = {'model': 'flashnext', 'prompt': prompt, 'max_tokens': 1, 'temperature': 0}
    async with session.post(url + '/v1/completions', json=payload) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {body[:500]}')
    usage = json.loads(body)['usage']
    if usage['prompt_tokens'] != len(prompt):
        raise RuntimeError('server did not consume the full prompt')
    return time.monotonic() - start


async def profiled(session, url, enabled, delay, duration, work):
    task = asyncio.create_task(work)
    if enabled:
        await asyncio.sleep(delay)
        if not task.done():
            async with session.post(url + '/start_profile') as r:
                r.raise_for_status()
            await asyncio.sleep(duration)
            async with session.post(url + '/stop_profile') as r:
                r.raise_for_status()
    return await task


async def main(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    root = Path(a.corpus_root)
    # One long stream of real source, cut into non-overlapping pieces so no request hits another's prefix.
    need = sum(a.cold) + a.depth + 8 * max(a.append) + 16
    stream = []
    for subdir, _ in CODEBASE_AGENTS:
        try:
            stream += codebase_ids(tok, root / subdir, need - len(stream))
        except RuntimeError:
            for path in sorted((root / subdir).rglob('*.py')):
                stream += tok.encode(path.read_text(errors='replace'), add_special_tokens=False, verbose=False)
        if len(stream) >= need:
            break
    if len(stream) < need:
        raise RuntimeError(f'corpus has {len(stream)} tokens, need {need}')
    cursor = 0

    def take(n):
        nonlocal cursor
        cursor += n
        return stream[cursor - n:cursor]

    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=a.timeout)) as session:
        for n in a.cold:
            prompt = take(n)
            prof = a.profile and n == max(a.cold)
            seconds = await profiled(session, a.url, prof, a.profile_delay, a.profile_seconds,
                                     complete(session, a.url, prompt))
            rows.append({'name': f'cold-{n}', 'tokens': n, 'seconds': seconds, 'tok_s': n / seconds, 'profiled': prof})
            print(json.dumps(rows[-1]), flush=True)
        context = take(a.depth)
        seconds = await complete(session, a.url, context)
        rows.append({'name': f'prime-{a.depth}', 'tokens': a.depth, 'seconds': seconds, 'tok_s': a.depth / seconds})
        print(json.dumps(rows[-1]), flush=True)
        for n in a.append:
            seconds = await complete(session, a.url, context + take(n))
            rows.append({'name': f'append-{a.depth}-{n}', 'tokens': n, 'seconds': seconds, 'tok_s': n / seconds})
            print(json.dumps(rows[-1]), flush=True)
        for n in a.append:
            prompts = [context + take(n) for _ in range(8)]
            prof = a.profile and n == max(a.append)

            async def all_eight():
                start = time.monotonic()
                times = await asyncio.gather(*(complete(session, a.url, p) for p in prompts))
                return time.monotonic() - start, times

            seconds, times = await profiled(session, a.url, prof, 1.0, a.profile_seconds, all_eight())
            rows.append({'name': f'append8-{a.depth}-{n}', 'tokens': 8 * n, 'seconds': seconds,
                         'tok_s': 8 * n / seconds, 'per_request_seconds': times, 'profiled': prof})
            print(json.dumps(rows[-1]), flush=True)
    Path(a.output).write_text(json.dumps({'rows': rows}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--corpus-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--cold', type=int, nargs='+', default=[16384, 65536, 200000])
    p.add_argument('--depth', type=int, default=200000)
    p.add_argument('--append', type=int, nargs='+', default=[4096, 12288])
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-delay', type=float, default=60)
    p.add_argument('--profile-seconds', type=float, default=4)
    p.add_argument('--timeout', type=int, default=7200)
    raise SystemExit(asyncio.run(main(p.parse_args())))
