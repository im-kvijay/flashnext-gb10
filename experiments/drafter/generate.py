"""Generate target continuations for drafter distillation.

Prompts come from a JSONL file ({"id", "prompt"}) that must not overlap the
speed or quality benchmarks (KodCode-Light-RL-10K, not HumanEval,
LiveCodeBench or the repository-source tasks). The official chat template
with thinking enabled and the checkpoint's sampling defaults are used, so the
continuations resemble real coding traffic. Writes one JSON line per request:
prompt token IDs and output token IDs.
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


async def main(a):
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    random.Random(a.seed).shuffle(rows)
    rows = rows[a.offset:a.offset + a.count]
    out = Path(a.output)
    done = {json.loads(line)['id'] for line in out.read_text().splitlines()} if out.exists() else set()
    queue = asyncio.Queue()
    for row in rows:
        if row['id'] not in done:
            queue.put_nowait(row)
    timeout = aiohttp.ClientTimeout(total=3600)
    lock = asyncio.Lock()
    produced = 0

    async def worker(session):
        nonlocal produced
        while not queue.empty():
            row = queue.get_nowait()
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                                 tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(text, add_special_tokens=False)
            payload = {'model': 'flashnext', 'prompt': prompt_ids, 'max_tokens': a.max_tokens,
                       'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'return_token_ids': True}
            async with session.post(a.url + '/v1/completions', json=payload) as response:
                body = await response.json()
                if response.status != 200:
                    raise RuntimeError(str(body)[:500])
            output_ids = body['choices'][0]['token_ids']
            async with lock:
                with out.open('a') as f:
                    f.write(json.dumps({'id': row['id'], 'prompt_token_ids': prompt_ids,
                                        'output_token_ids': output_ids}) + '\n')
                produced += len(output_ids)
                print(f"{row['id']}: {len(prompt_ids)} + {len(output_ids)} tokens (total {produced})", flush=True)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(worker(session) for _ in range(a.concurrency)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--prompts', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=160)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--concurrency', type=int, default=8)
    asyncio.run(main(p.parse_args()))
