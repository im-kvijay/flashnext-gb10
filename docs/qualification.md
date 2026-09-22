# Release evidence required

The release target is eight agents on one GB10, each with at least 200,000
tokens of usable context. A shorter diagnostic run cannot satisfy this target.
The minimum aggregate output throughput is 400 tokens/second at the required
concurrency and context. Drafted-but-rejected tokens do not count as output.

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
- Drafting is measured by accepted tokens, verification cost, and end-to-end
  throughput. Compare MTP depths and drafter alternatives; verify target-model
  sampling semantics rather than assuming a high acceptance rate proves quality.

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

The warm fixed-length eager test generated 2048 tokens in 24.701 seconds, or
82.912 aggregate tokens/s including prefill, with all eight streams overlapping.
This test used `ignore_eos` to hold a fixed budget and is explicitly a synthetic
stress result. It does not qualify natural workload throughput.

The same baseline passed the single-agent 200,000-input-token retrieval probe,
returning 195 output tokens and the correct code. TTFT was 91.878 seconds and
wall time 103.135 seconds. It proves that one long request worked; it does not
prove eight concurrent long requests or broad long-context reasoning.

The graph/FP8-KV/native-MTP-3 candidate passed 8/8 short retrieval probes and
reached 147.966 output tokens/s during the 10.827-second interval when all eight
streams were generating in the synthetic fixed-output test. End-to-end throughput
was 76.811 tokens/s, below the eager baseline. It reserved 21.43 GiB for KV and
reported capacity for 1,229,544 tokens, still insufficient for eight long agents.

Its first 200k probe exposed the client's SSE line-size limit: requesting token
IDs makes vLLM echo a large prompt-ID array. The client now sizes its read buffer
for the declared context. The corrected retry passed with 308 output tokens in
10.961 seconds, TTFT 2.695 seconds. This reused the prior processed prompt and
is a warm-prefix result, not a new cold-prefill speed measurement.

The draft-only vocabulary implementation passed 16 changing CUDA graph replays
with eight rows each, matching the corresponding subset of full-head logits.
Full-model throughput and retention for this optimization remain unmeasured.
The text-only candidate omits the vision encoder and does not support image or
video inputs; this tradeoff must remain explicit if it becomes a release option.

The text-only/FP8-KV/reduced-draft candidate at memory fraction 0.88 passed 8/8
short retrievals and reached 167.924 tokens/s in the all-stream synthetic decode
interval. It reported 28.78 GiB KV and 1,650,688 tokens. During its 200k screen,
SSH stopped responding and Vast reported the host offline. The last available
memory reading was about 5.4 GiB. Memory pressure is suspected, not confirmed;
no final long-screen receipt or kernel diagnosis was retrieved. A Vast container
reboot was requested. This is a reliability failure, not a qualified capacity win.

The default memory fraction is reduced to 0.80 pending further measurement.
`scripts/supervise.py` records MemAvailable every second and terminates only its
own service process group after a sustained breach of the 8 GiB floor. Use it
for subsequent hardware experiments. The floor is a guard, not proof against
driver or provider failures. Packed PLE state and wider verification graphs are
staged but have not yet been measured in a full-model run.

Reconstructing the recorded cache layout through the pinned upstream allocator
passed: normalizing the PLE replication marker at TP=DP=PP=1 reduces six cache
groups to five, and 88 blocks per 212,992-token request to 83. Every layer remains
present; all state dimensions, dtypes and rollback counts remain unchanged. The
plugin rejects multi-device configurations and KV connectors. Full-model
continuation/rollback tests are still required.

On the replacement GB10, the packed-state/MTP-2/FP8-KV candidate with all
verification graph sizes captured passed 8/8 short retrievals. Its synthetic
eight-stream overlap rate was 102.688 tokens/s (43.377 including prefill), with
no preemptions and 1284/1528 draft tokens accepted. This host was observed at
604 MHz under active inference; a separate short BF16 GEMM probe stayed near
598 MHz. The driver refused a reset to default clocks. These observations do
not isolate clocks from the changed host, MTP depth, and memory configuration.

An explicit 8 GiB cache reported 505,856 tokens of capacity (2.38 requests at
212,992). The single 200k probe was terminated by the supervisor at 7.699 GiB
MemAvailable before producing an answer. The host remained reachable. This
is a recorded memory-floor failure, not a successful long-context result.

Optional `b12x==1.3.0` passed the pinned upstream small NVFP4 MoE numerical
reference and 16 changing graph replays at each of 1, 8, and 24 tokens. Eager
and replay outputs matched exactly in those fixtures. Testing used CuTe DSL
4.7.1 despite the package declaring 4.6.2; this is an experimental combination,
not a default dependency change or a general compatibility claim. Full-model
quality, throughput, and memory remain to be measured. Use the target's B12x
MoE backend with the draft backend explicitly set to `auto`, because the
NVIDIA checkpoint's MTP experts are FP8 rather than NVFP4.
