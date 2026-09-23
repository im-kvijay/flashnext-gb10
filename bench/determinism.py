"""Run-to-run determinism of teacher-forced scoring on one server.

Scores the same fidelity sequences several times against one running server:
alone with identical scheduling, and paired with another request. Differences
between solo repeats point to a race or an unordered reduction; differences
that appear only when paired point to batch-composition sensitivity.
"""
import argparse
import asyncio
import json
import math
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from fidelity import score, sequences


def diff(a, b):
    n = len(a['rows'])
    big = agree = 0
    worst = total = 0.0
    identical = 0
    for (a_real, a_top), (b_real, b_top) in zip(a['rows'], b['rows']):
        d = abs(a_real - b_real)
        total += d
        worst = max(worst, d)
        big += d > 0.5
        identical += a_top == b_top
        agree += max(a_top, key=a_top.get) == max(b_top, key=b_top.get)
    return dict(positions=n, identical_topk=identical / n, top1_agreement=agree / n,
                mean_abs_dlogp=total / n, max_abs_dlogp=worst, frac_dlogp_gt_05=big / n)


async def run(a):
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    seqs = {s['name']: s for s in sequences(tokenizer, a)}
    timeout = aiohttp.ClientTimeout(total=7200)
    report = {}
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=1 << 24) as session:
        async def solo(name):
            return await score(session, a.url, seqs[name], a.top_k)

        async def solo_seq(seq):
            return await score(session, a.url, seq, a.top_k)

        for name in ('codebase-0', 'workload-0'):
            runs = [await solo(name) for _ in range(a.repeats)]
            report[f'{name} solo 1 vs 2'] = diff(runs[0], runs[1])
            if a.repeats > 2:
                report[f'{name} solo 1 vs 3'] = diff(runs[0], runs[2])
            other = name[:-1] + '1'
            paired, _ = await asyncio.gather(solo(name), solo(other))
            report[f'{name} solo vs paired'] = diff(runs[0], paired)
            nll = -sum(r[0] for r in runs[0]['rows']) / len(runs[0]['rows'])
            report[f'{name} mean nll'] = nll
            runs[0]['ids'] = seqs[name]['ids']
            report.setdefault('_records', []).append(runs[0])
        # Context monotonicity: a healthy model never finds the same span less
        # likely with more preceding context.
        ids = seqs['codebase-0']['ids']
        for offset in (512, 1024, 1536, 2048):
            span = ids[offset:offset + 256]
            full = dict(name='full', ids=ids[:offset + 256], score_from=offset)
            short_ids = ids[:32] + ids[offset - 128:offset + 256]
            short = dict(name='short', ids=short_ids, score_from=len(short_ids) - len(span))
            f, s = await asyncio.gather(solo_seq(full), solo_seq(short))
            report[f'codebase-0 span@{offset} nll full/short'] = [
                -sum(r[0] for r in f['rows']) / len(span), -sum(r[0] for r in s['rows']) / len(span)]
    records = report.pop('_records')
    Path(a.output).write_text(json.dumps(dict(report=report, records=records)))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--source', help='concurrency.py workload result with recorded output token IDs')
    p.add_argument('--corpus-root')
    p.add_argument('--agents', type=int, default=2)
    p.add_argument('--scored-tokens', type=int, default=2048)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output')
    asyncio.run(run(p.parse_args()))
