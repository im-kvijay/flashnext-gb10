"""Count distinct routed experts per decode step for eight concurrent agents.

Requires a server started with --enable-return-routed-experts. Each agent gets a
different natural engineering task. Routing of the generated tokens is replayed
as a concurrent batch: at every step each agent contributes a window of k+1
consecutive positions (the verified span for k draft tokens), advancing by the
measured mean accepted length. Rejected drafts are approximated by the tokens
that actually occupied those positions. Output: distinct experts per layer per
step, which sets the routed-expert bytes read per step.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer

TASKS = [
    "Implement a thread-safe LRU cache in Python with TTL expiry and size-based eviction, with unit tests.",
    "Write a Rust function that parses a subset of TOML (tables, strings, integers, arrays) into a map, with error reporting by line.",
    "Explain and fix a deadlock in a Go worker pool where workers and the dispatcher both block on unbuffered channels. Provide corrected code.",
    "Design a PostgreSQL schema and queries for a multi-tenant invoicing system with partial payments and refunds. Explain indexing choices.",
    "Write a TypeScript React hook for paginated infinite scrolling with request cancellation and error retry, plus tests.",
    "Implement Dijkstra and A* in C++ for a weighted grid with obstacles; compare their complexity and show a benchmark harness.",
    "Refactor a 300-line Flask view that mixes validation, database access and templating into services with dependency injection; show the result.",
    "Write a bash script that rotates logs by size and age, compresses old files, and is safe under concurrent invocation; explain the locking.",
]


def decode_routing(value, output_tokens, layers, top_k):
    raw = np.frombuffer(base64.b64decode(value), dtype=np.uint16)
    rows = raw.reshape(-1, layers, top_k)
    return rows[-output_tokens:]


async def one(session, url, prompt_ids, max_tokens):
    payload = {"model": "flashnext", "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
               "return_token_ids": True, "routed_experts_prompt_start": len(prompt_ids)}
    async with session.post(url + "/v1/completions", json=payload) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(str(body)[:1000])
    choice = body["choices"][0]
    return dict(text=choice["text"], finish=choice["finish_reason"],
                token_ids=choice.get("token_ids") or [], routed=choice.get("routed_experts"))


def replay(routes, window, stride, layers):
    """Mean distinct experts per layer per step across all agents."""
    counts = []
    positions = [0.0] * len(routes)
    while all(int(p) + window <= len(r) for p, r in zip(positions, routes)):
        per_layer = np.zeros(layers)
        for layer in range(layers):
            experts = set()
            for p, r in zip(positions, routes):
                experts.update(r[int(p):int(p) + window, layer].ravel().tolist())
            per_layer[layer] = len(experts)
        counts.append(per_layer.mean())
        positions = [p + stride for p in positions]
    return float(np.mean(counts)), len(counts)


async def main(a):
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    config = json.load(open(Path(a.model) / "config.json"))
    text = config.get("text_config", config)
    layers, experts, top_k = text["num_hidden_layers"], text["num_experts"], text["num_experts_per_tok"]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                             add_generation_prompt=True) for t in TASKS[:a.agents]]
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*(one(session, a.url, ids, a.output_tokens) for ids in prompt_ids))
    routes = []
    for r in results:
        if not r["routed"]:
            raise RuntimeError("server did not return routed experts; start it with --enable-return-routed-experts")
        routes.append(decode_routing(r["routed"], len(r["token_ids"]), layers, top_k))
    report = dict(scope="routing replay estimate; not a throughput measurement", agents=a.agents,
                  output_tokens=[len(r) for r in routes], finishes=[r["finish"] for r in results],
                  accepted_length=a.accepted_length, rows=[])
    for k in range(0, 8):
        window = k + 1
        stride = min(a.accepted_length, window) if k else 1
        mean, steps = replay(routes, window, stride, layers)
        uniform = experts * (1 - (1 - top_k / experts) ** (a.agents * window))
        report["rows"].append(dict(draft_tokens=k, tokens_per_step=a.agents * window, steps=steps,
                                   distinct_experts_per_layer=mean, uniform_estimate=uniform,
                                   fraction_of_uniform=mean / uniform))
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    for row in report["rows"]:
        print(json.dumps(row))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--output-tokens", type=int, default=1024)
    p.add_argument("--accepted-length", type=float, default=2.18)
    p.add_argument("--output", required=True)
    asyncio.run(main(p.parse_args()))
