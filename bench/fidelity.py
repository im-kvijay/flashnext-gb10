"""Teacher-forced next-token fidelity of a serving configuration.

Replays fixed sequences (a prompt plus a recorded model continuation, or real
source code) and records, at every scored position, the server's top-k
log-probabilities and the log-probability of the realized token. Two records
from different configurations are compared with --compare: top-1 agreement,
approximate KL(reference || candidate) on the reference's top-k support, and
the mean change in realized-token NLL. This detects small numerical changes
(weight or KV precision) far more sensitively than a small benchmark.
"""
import argparse
import asyncio
import json
import math
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from concurrency import make_prompt


def sequences(tokenizer, a):
    seqs = []
    source = json.loads(Path(a.source).read_text())
    for row in source['requests'][:a.agents]:
        prompt, _ = make_prompt(tokenizer, row['input_tokens'], row['agent'], 'workload')
        seqs.append(dict(name=f"workload-{row['agent']}", ids=prompt + row['output_token_ids'][:a.scored_tokens],
                         score_from=len(prompt)))
    for agent in range(a.agents):
        ids, _ = make_prompt(tokenizer, a.scored_tokens + 256, agent, 'codebase', a.corpus_root)
        seqs.append(dict(name=f'codebase-{agent}', ids=ids, score_from=256))
    return seqs


async def score(session, url, seq, top_k):
    payload = {'model': 'flashnext', 'prompt': seq['ids'], 'max_tokens': 1, 'temperature': 0,
               'prompt_logprobs': top_k}
    async with session.post(url + '/v1/completions', json=payload) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(str(body)[:500])
    entries = body['choices'][0]['prompt_logprobs']
    rows = []
    for position in range(seq['score_from'], len(seq['ids'])):
        entry = entries[position]
        token = str(seq['ids'][position])
        top = {k: v['logprob'] for k, v in entry.items() if v.get('rank', 99) <= top_k}
        rows.append([entry[token]['logprob'], top])
    return dict(name=seq['name'], rows=rows)


async def run(a):
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    seqs = sequences(tokenizer, a)
    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=1 << 24) as session:
        # Two at a time bounds the server's prompt-logprob memory.
        results = []
        for i in range(0, len(seqs), 2):
            results += await asyncio.gather(*(score(session, a.url, s, a.top_k) for s in seqs[i:i + 2]))
    Path(a.output).write_text(json.dumps(dict(top_k=a.top_k, sequences=results)))
    print(json.dumps({'sequences': len(results), 'scored_positions': sum(len(r['rows']) for r in results)}))


def compare(reference_path, candidate_path):
    ref = {s['name']: s['rows'] for s in json.loads(Path(reference_path).read_text())['sequences']}
    cand = {s['name']: s['rows'] for s in json.loads(Path(candidate_path).read_text())['sequences']}
    report = {}
    for group in ('workload', 'codebase', 'all'):
        agree = n = 0
        kl = dnll = 0.0
        for name in ref:
            if name not in cand or (group != 'all' and not name.startswith(group)):
                continue
            for (r_real, r_top), (c_real, c_top) in zip(ref[name], cand[name]):
                n += 1
                agree += max(r_top, key=r_top.get) == max(c_top, key=c_top.get)
                floor = min(c_top.values())
                rz = sum(math.exp(v) for v in r_top.values())
                kl += sum(math.exp(v) / rz * (v - c_top.get(k, floor)) for k, v in r_top.items())
                dnll += r_real - c_real
        if n:
            report[group] = dict(positions=n, top1_agreement=agree / n, approx_kl=kl / n,
                                 mean_nll_increase=dnll / n)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--source', help='concurrency.py workload result with recorded output token IDs')
    p.add_argument('--corpus-root')
    p.add_argument('--agents', type=int, default=8)
    p.add_argument('--scored-tokens', type=int, default=2048)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--output')
    p.add_argument('--compare', nargs=2, metavar=('REFERENCE', 'CANDIDATE'))
    a = p.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=2))
    else:
        asyncio.run(run(a))
