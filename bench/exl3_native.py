"""Native EXL3 screen using the same exact token prompts as the vLLM screen.

This bypasses HTTP and tool parsing. It is engine evidence, not API qualification.
Stop tokens are respected; only emitted token IDs count toward throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Set before importing the fork; avoid its default target HC int8 conversion.
os.environ["EXL3_GR_INT8"] = "0"
os.environ["EXL3_MTP_HEAD_N"] = "0"
os.environ["EXL3_NGRAM_STREAM"] = "1"

import torch
from transformers import AutoTokenizer
from exllamav3 import (Config, Model, Cache, CacheLayer_quant, Tokenizer,
                      Generator, Job, GreedySampler)
from concurrency import make_prompt


def main(a):
    output = Path(a.output)
    if output.exists():
        raise ValueError("Refusing to replace an existing experiment")
    output.parent.mkdir(parents=True, exist_ok=True)
    hf = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = [make_prompt(hf, a.input_tokens, i, a.mode) for i in range(a.concurrency)]
    required = a.concurrency * ((a.input_tokens + a.output_tokens + a.mtp + 255) // 256 * 256)
    if a.cache_tokens < required:
        raise ValueError(f"Cache needs at least {required} tokens for all requests")
    start_load = time.monotonic()
    config = Config.from_directory(a.model)
    config.infer_params.ngram_stream_from_disk = True
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    # Check native and HF token identities on the actual full prompts before loading.
    for ids, _ in prompts:
        text = hf.decode(ids, skip_special_tokens=False)
        native = tokenizer.encode(text, encode_special_tokens=True).flatten().tolist()
        if native != hf.encode(text, add_special_tokens=False):
            raise ValueError("EXL3 and official tokenizer disagree")
    cache = Cache(model, max_num_tokens=a.cache_tokens, layer_type=CacheLayer_quant,
                  k_bits=8, v_bits=8, max_batch_size=a.concurrency, max_history=a.mtp)
    draft = draft_cache = None
    if a.mtp:
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=a.cache_tokens,
            layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
            max_batch_size=a.concurrency, max_history=a.mtp)
        draft.load(device=torch.device("cuda:0"), progressbar=True)
    model.load(device=torch.device("cuda:0"), progressbar=True)
    generator = Generator(model, cache, tokenizer, max_batch_size=a.concurrency,
        max_chunk_size=a.prefill, draft_model=draft, draft_cache=draft_cache,
        num_draft_tokens=a.mtp, recurrent_cache_size=a.recurrent_cache_gib * 1024**3)
    loaded_seconds = time.monotonic() - start_load
    stops = set(config.eos_token_id_list)
    if not stops:
        raise ValueError("Missing EOS token IDs")
    jobs = []
    rows = []
    for i, (ids, expected) in enumerate(prompts):
        jobs.append(Job(input_ids=torch.tensor([ids], dtype=torch.long),
            max_new_tokens=a.output_tokens, sampler=GreedySampler(),
            stop_conditions=stops, identifier=i, decode_special_tokens=True))
        rows.append(dict(agent=i, input_tokens=len(ids), expected=expected,
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            token_events=[], output_token_ids=[], text="", finish=None, requeues=0))
    started = time.monotonic()
    generator.enqueue(jobs)
    while generator.num_remaining_jobs():
        events = generator.iterate()
        now = time.monotonic()
        for event in events:
            row = rows[event["job"].identifier]
            ids = event.get("token_ids")
            if ids is not None and ids.numel():
                ids = ids.flatten().tolist()
                row["token_events"].append([now, len(ids)])
                row["output_token_ids"].extend(ids)
            row["text"] += event.get("text", "")
            row["requeues"] += bool(event.get("requeue"))
            if event.get("eos"):
                assert event["prompt_tokens"] == a.input_tokens
                row["finish"] = event["eos_reason"]
                row["end"] = now
                row["native_counters"] = {k: event[k] for k in (
                    "new_tokens", "prompt_tokens", "time_prefill", "time_generate",
                    "cached_tokens", "accepted_draft_tokens", "rejected_draft_tokens") if k in event}
                answer = row["text"].split("</think>")[-1].strip()
                row["retrieval_correct"] = a.mode == "retrieval" and answer == row["expected"]
                print(json.dumps({"agent": row["agent"], "finish": row["finish"],
                    "emitted_tokens": len(row["output_token_ids"]),
                    "retrieval_correct": row["retrieval_correct"]}), flush=True)
    elapsed = time.monotonic() - started
    assert all(row["finish"] and row["token_events"] for row in rows)
    overlap_start = max(row["token_events"][0][0] for row in rows)
    overlap_end = min(row["token_events"][-1][0] for row in rows)
    overlap = max(0, overlap_end - overlap_start)
    tokens = sum(n for row in rows for t, n in row["token_events"] if overlap_start < t <= overlap_end)
    summary = dict(engine="exl3_native", mode=a.mode, concurrency=a.concurrency,
        input_tokens_each=a.input_tokens, mtp=a.mtp, load_seconds=loaded_seconds,
        wall_seconds=elapsed, all_streams_overlap_seconds=overlap,
        all_streams_overlap_output_tokens=tokens,
        all_streams_overlap_output_tps=tokens / overlap if overlap else None,
        aggregate_output_tps_including_prefill=sum(len(r["output_token_ids"]) for r in rows) / elapsed,
        naturally_finished_requests=sum(r["finish"] == "stop_token" for r in rows),
        retrieval_correct=sum(r["retrieval_correct"] for r in rows),
        requeues=sum(r["requeues"] for r in rows), release_qualified=False)
    output.write_text(json.dumps(dict(summary=summary, requests=rows, arguments=vars(a)), indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True, help="Official NVIDIA tokenizer path")
    p.add_argument("--output", required=True)
    p.add_argument("--input-tokens", type=int, default=4096)
    p.add_argument("--output-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cache-tokens", type=int, default=262144)
    p.add_argument("--prefill", type=int, default=2048)
    p.add_argument("--recurrent-cache-gib", type=int, default=4)
    p.add_argument("--mtp", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--mode", choices=("retrieval", "workload"), default="retrieval")
    main(p.parse_args())
