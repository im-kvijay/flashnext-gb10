# Release evidence required

The release target is eight agents on one GB10, each with at least 200,000
tokens of usable context. A shorter diagnostic run cannot satisfy this target.

The tests must establish:

- Full NVIDIA model and n-gram table retained, with a verified pinned checkpoint.
- Exact PLE storage and lookup bytes through CPU gather and CUDA graph replay.
- Eight distinct 200k input contexts consumed without truncation, eviction loops,
  request errors, stale PLE state, or output degeneration; actual output overlap.
- Successful continuing tool calls using the official chat template and parser.
- Quality comparison for any new weight/KV quantization, across reasoning, coding,
  structured instructions, tools, and long-context retrieval. Retrieval alone
  does not establish general intelligence retention.
- Repeated aggregate throughput, time to first token, inter-token latency, and
  wall time, including failed attempts. Cold and warm results are separate.
- A clean clone on GB10 can install the pinned runtime, download and verify the
  checkpoint, start the service and repeat qualification.
- Kernel/graph optimizations have measured benefit and numerical validation.

## Current evidence

The initial implementation passed 32 changing FP8 table lookup batches on GB10,
with eight rows of agent inputs per batch, duplicate indices and masked indices.
It also passed 32 CUDA graph replays with changing inputs across eight agents.
The first implementation allocated pinned memory inside a background worker;
the replay test caught capture invalidation. Staging is now preallocated before
capture, and the changing-input replay test passes.

The first real-model eager/BF16-KV screen passed 8/8 retrieval requests at 4096
input tokens each. All eight output streams overlapped. Total elapsed time was
31.739 seconds and 1192 output tokens were generated: 37.557 output tokens/s
including prefill. This is one startup and one run, with first-use JIT overhead;
it is not the optimized throughput result and does not meet the context target.

At memory fraction 0.88 the engine reports 28.11 GiB KV capacity, 1,114,902
tokens, or 5.23 requests at 212,992 tokens each. The baseline does not meet the
eight-agent capacity requirement. Do not label it production-ready.

NVMe storage was measured at approximately 83,468 random 4 KiB reads/s at queue
depth 16. This storage measurement does not prove negligible model latency.

Full-model benchmarks are running; release status remains unqualified.
