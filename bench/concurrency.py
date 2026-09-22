"""Concurrent real-context stress and retrieval test, with streamed timing.

The full prompt and output token counts are verified from server usage. Exact
input token IDs prevent character-count approximations. Every request is distinct.
Performance mode forces generation and makes no quality claim; retrieval mode
allows normal EOS and requires the request-specific answer.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time

import aiohttp
from transformers import AutoTokenizer


def make_prompt(tokenizer, count, agent):
    rng = random.Random(7919 + agent)
    secret = f"{rng.randrange(10**9, 10**10)}"
    prefix = tokenizer.encode(f"Agent {agent}: independent record collection.\n", add_special_tokens=False)
    # Distinct realistic vocabulary and numbers exercise more PLE rows than a
    # single repeated filler token. The marker is placed at a different depth.
    lines = []
    while len(lines) < count // 12 + 100:
        lines.append(f"Record {rng.randrange(10**8)}: service {rng.choice(['parser','cache','scheduler','storage','worker'])} "
                     f"completed batch {rng.randrange(10000)} with checksum {rng.randrange(10**10)}.\n")
    filler = tokenizer.encode("".join(lines), add_special_tokens=False, verbose=False)
    marker = tokenizer.encode(f"\nThe verification code for agent {agent} is {secret}.\n", add_special_tokens=False)
    query = f"Return only the verification code for agent {agent} stated in the records. Do not use any other agent's code."
    # Tokenize the official chat template with a placeholder then replace its
    # content token interval. No custom template or thinking suppression.
    sentinel = "FLASHNEXT_UNIQUE_USER_CONTENT_SENTINEL_723190"
    empty_text = tokenizer.apply_chat_template([{"role":"user","content":sentinel}], tokenize=False, add_generation_prompt=True)
    if empty_text.count(sentinel) != 1:
        raise RuntimeError("cannot locate the user-content interval in the official template")
    split = empty_text.index(sentinel)
    before = tokenizer.encode(empty_text[:split], add_special_tokens=False)
    after = tokenizer.encode("\n" + query + empty_text[split + len(sentinel):], add_special_tokens=False)
    needed = count - len(before) - len(after) - len(prefix) - len(marker)
    if needed < 0:
        raise ValueError("prompt length too small")
    if len(filler) < needed:
        raise RuntimeError("insufficient unique filler tokens")
    position = int(needed * ((agent % 8 + 0.5) / 8))
    ids = before + prefix + filler[:position] + marker + filler[position:needed] + after
    assert len(ids) == count
    return ids, secret


async def run_one(session, url, index, prompt, expected, args, barrier):
    await barrier.wait()
    start = time.monotonic()
    result = {"agent": index, "expected": expected, "start": start,
              "input_tokens":len(prompt), "input_sha256":hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
              "first_token":None, "usage":None, "text":"", "finish_reason":None, "error":None}
    payload = {"model":"flashnext", "prompt":prompt, "max_tokens":args.output_tokens,
               "temperature":0, "stream":True, "stream_options":{"include_usage":True}}
    if args.mode == "performance":
        payload.update(ignore_eos=True, min_tokens=args.output_tokens)
    try:
        async with session.post(url + "/v1/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    text = choice.get("text", "")
                    if text and result["first_token"] is None:
                        result["first_token"] = time.monotonic()
                    result["text"] += text
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
        if not result["usage"] or result["usage"]["prompt_tokens"] != len(prompt):
            raise RuntimeError("missing usage or server did not consume the full context")
        if not result["text"] or result["usage"]["completion_tokens"] <= 0:
            raise RuntimeError("empty output")
        if args.mode == "performance" and result["usage"]["completion_tokens"] != args.output_tokens:
            raise RuntimeError("server did not generate the required performance token budget")
    except Exception as exc:
        result["error"] = str(exc)
    result["end"] = time.monotonic()
    result["wall_seconds"] = result["end"] - start
    result["ttft_seconds"] = None if result["first_token"] is None else result["first_token"] - start
    answer = result["text"].split("</think>")[-1].strip()
    result["retrieval_correct"] = args.mode == "retrieval" and answer == expected
    print(json.dumps({k:result[k] for k in ['agent','wall_seconds','ttft_seconds','usage','error','retrieval_correct']}), flush=True)
    return result


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [make_prompt(tokenizer, args.input_tokens, i) for i in range(args.concurrency)]
    barrier = asyncio.Event()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        tasks = [asyncio.create_task(run_one(session,args.url,i,p,s,args,barrier)) for i,(p,s) in enumerate(prompts)]
        barrier.set()
        results = await asyncio.gather(*tasks)
    wall = max(r['end'] for r in results) - min(r['start'] for r in results)
    successful_tokens = sum((r['usage'] or {}).get('completion_tokens',0) for r in results if not r['error'])
    events = []
    for r in results:
        if r['first_token'] is not None:
            events.extend([(r['first_token'],1),(r['end'],-1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak,active)
    summary = {'mode':args.mode,'concurrency':args.concurrency,'input_tokens_each':args.input_tokens,
               'output_tokens_limit':args.output_tokens,'wall_seconds':wall,
               'aggregate_output_tps_including_prefill':successful_tokens/wall,
               'peak_overlapping_output_streams':peak,
               'errors':sum(r['error'] is not None for r in results),
               'retrieval_correct':sum(r['retrieval_correct'] for r in results),
               'release_qualified':False}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps({'summary':summary,'requests':results},indent=2))
    print(json.dumps(summary,indent=2))
    return 1 if summary['errors'] else 0


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--model',required=True)
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--input-tokens',type=int,default=200000)
    p.add_argument('--output-tokens',type=int,default=8192)
    p.add_argument('--mode',choices=['retrieval','performance'],default='retrieval')
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if min(args.concurrency,args.input_tokens,args.output_tokens,args.timeout) <= 0:
        p.error('counts and timeout must be positive')
    raise SystemExit(asyncio.run(main(args)))
