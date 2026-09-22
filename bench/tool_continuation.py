"""Independent agents call a tool, consume its result, and return their own code."""
import argparse
import asyncio
import json
from pathlib import Path
import random
import time

import aiohttp
from transformers import AutoTokenizer


TOOLS = [{"type": "function", "function": {
    "name": "lookup_record", "description": "Retrieve a record's verification code.",
    "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}},
                   "required": ["record_id"], "additionalProperties": False},
}}]


def prepare(tokenizer, minimum_tokens, agent):
    rng = random.Random(17291 + agent)
    record = f"agent_{agent}_record"
    code = f"AGENT{agent}_CODE_{rng.randrange(10**11, 10**12)}"
    lines = [f"Record {rng.randrange(10**9)}: worker {rng.randrange(1000)} completed batch {rng.randrange(10**8)}.\n"
             for _ in range(minimum_tokens // 8 + 512)]
    filler = tokenizer.encode("".join(lines), add_special_tokens=False, verbose=False)
    n = minimum_tokens
    for _ in range(12):
        body = tokenizer.decode(filler[:n], skip_special_tokens=False)
        messages = [
            {"role": "system", "content": "Use lookup_record when asked for a verification code. Never invent a tool result."},
            {"role": "user", "content": f"Agent {agent} context records:\n{body}\n"
             f"Call lookup_record exactly once with record_id {record}. The verification code is available only from that tool."},
        ]
        rendered = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(rendered, add_special_tokens=False, verbose=False))
        if minimum_tokens <= count <= minimum_tokens + 16:
            assert code not in messages[1]["content"]
            return messages, record, code, count
        n += minimum_tokens - count
        if not 0 <= n <= len(filler):
            raise ValueError("Requested context cannot fit the tool template")
    raise RuntimeError("Unable to construct the requested tool context")


async def run_one(session, args, index, prepared, barrier):
    messages, record, code, template_tokens = prepared
    result = {"agent": index, "record_id": record, "expected_code": code,
              "template_tokens": template_tokens, "responses": [], "tool_call_valid": False,
              "continuation_correct": False, "error": None}
    await barrier.wait()
    started = time.monotonic()

    async def completion(messages, tool_choice):
        payload = {"model": "flashnext", "messages": messages, "tools": TOOLS,
                   "tool_choice": tool_choice, "temperature": 0, "max_tokens": args.output_tokens}
        async with session.post(args.url + "/v1/chat/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
            value = await response.json()
        result["responses"].append(value)
        usage = value.get("usage") or {}
        if usage.get("prompt_tokens", 0) < args.input_tokens:
            raise RuntimeError("Server did not attest the minimum input context")
        if usage.get("completion_tokens", 0) <= 0:
            raise RuntimeError("Server returned no generated tokens")
        return value["choices"][0]

    try:
        first = await completion(messages, "auto")
        assistant = first["message"]
        calls = assistant.get("tool_calls") or []
        if first.get("finish_reason") != "tool_calls" or len(calls) != 1:
            raise RuntimeError("Expected exactly one completed tool call")
        call = calls[0]
        function = call["function"]
        if function["name"] != "lookup_record" or json.loads(function["arguments"]) != {"record_id": record}:
            raise RuntimeError("Tool call selected the wrong function or another agent's record")
        result["tool_call_valid"] = True
        # Send only portable chat fields, excluding backend-specific response metadata.
        messages = messages + [{"role": "assistant", "content": assistant.get("content"), "tool_calls": calls},
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"record_id": record, "verification_code": code})},
            {"role": "user", "content": "Return only the verification_code from that tool result, with no explanation."}]
        second = await completion(messages, "none")
        answer = (second["message"].get("content") or "").split("</think>")[-1].strip()
        result["continuation_correct"] = second.get("finish_reason") == "stop" and answer == code
    except Exception as exc:
        result["error"] = str(exc)
    result["wall_seconds"] = time.monotonic() - started
    print(json.dumps({k: result[k] for k in ("agent", "tool_call_valid", "continuation_correct", "error", "wall_seconds")}), flush=True)
    return result


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if len(tokenizer) < 1000 or not tokenizer.encode("Tokenizer readiness check", add_special_tokens=False):
        raise RuntimeError("Tokenizer is incomplete; finish the pinned checkpoint download before testing")
    prepared = [prepare(tokenizer, args.input_tokens, i) for i in range(args.concurrency)]
    if args.prepare_only:
        print(json.dumps({"template_tokens": [x[3] for x in prepared], "distinct_records": len({x[1] for x in prepared})}))
        return 0
    barrier = asyncio.Event()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        tasks = [asyncio.create_task(run_one(session, args, i, row, barrier)) for i, row in enumerate(prepared)]
        barrier.set()
        results = await asyncio.gather(*tasks)
    summary = {"concurrency": args.concurrency, "minimum_input_tokens": args.input_tokens,
               "valid_tool_calls": sum(x["tool_call_valid"] for x in results),
               "correct_continuations": sum(x["continuation_correct"] for x in results),
               "errors": sum(x["error"] is not None for x in results), "release_qualified": False}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "requests": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(summary["correct_continuations"] != args.concurrency or summary["errors"] > 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--input-tokens", type=int, default=200000)
    p.add_argument("--output-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--output", required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if min(args.concurrency, args.input_tokens, args.output_tokens, args.timeout) <= 0:
        p.error("Counts and timeout must be positive")
    raise SystemExit(asyncio.run(main(args)))
