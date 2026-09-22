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

The September 22 cheaper-host run (`cheaper-direct-mtp2`) loaded the complete
NVIDIA target and native MTP head with direct-checkpoint PLE, FP8 KV, packed PLE
state, a 29,977-token draft-only vocabulary, and all MTP-2 verification graph
sizes. Loading used 75.54 GiB and took 900.62 seconds; graph capture added
0.60 GiB. Its explicit 6 GiB cache is a diagnostic budget, reporting 380,723
tokens or 1.79 requests at 212,992 tokens, not eight long requests.

It passed 8/8 distinct 4k retrievals with natural stops. All-eight overlap
throughput was 127.24 output tokens/s (47.63 including prefill). The fixed-output
stress run reached 134.29 during overlap (72.23 including prefill), with 1277 of
1544 draft tokens accepted. Both had zero preemptions.

The coding-workload run generated 32,768 normal pre-EOS tokens, reaching its
4096-token output limit on all eight requests. Its profiled all-eight overlap
rate was 93.02 tokens/s and end-to-end rate 87.23. The bounded profiler perturbs
timing; this is diagnostic evidence, not a release throughput measurement.
There were no request errors or preemptions. These truncated outputs do not
establish completion or correctness of the requested coding tasks. The trace
showed 81.3% GPU event coverage, with the two routed-expert GEMMs accounting for
about 38% of the measured interval. Dense projections and GDN updates also
contribute substantially.

The following retention attempt initially failed in the client because
Transformers 5 returned `BatchEncoding` from template tokenization. Explicit
template rendering followed by encoding fixes the client path. The resumed
screen scored 10/12 with no request errors, consuming 20,225 output tokens in
241.58 seconds. Both misses were permutation-counting cases that exhausted the
4096-token output budget, so the original responses and limits are retained.
The eight-agent tool check passed 8/8 parsed calls and 8/8 correct continuations.
The cold single-agent 200k retrieval passed with 313 output tokens, 95.05 seconds
TTFT and 104.01 seconds wall time. None of these proves eight simultaneous 200k
contexts. Full release qualification remains absent.

## Earlier experiments and failure evidence

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

That B12x full-model startup subsequently failed during graph warmup with an
illegal CUDA memory access, before the service became ready. The asynchronous
error surfaced in HC SiLU; the originating kernel has not been localized.
Isolated checks at the actual expert geometry (512 experts, hidden 2560,
intermediate 640, top-10 routing) then passed at prefill size 2048 and changing
graph replay sizes 1 through 9, 12, 15, 18, 21, and 24. These checks do not
establish correctness of the full multilayer integration or real checkpoint
scales. B12x remains experimental and is not enabled by default.

The optional direct-checkpoint PLE backend (`FLASHNEXT_PLE_DIRECT=1`) gathers
the original FP8 bytes from retained safetensors mappings rather than copying
the entire table to a second file. It passed all 256 byte patterns, invalid
IDs, shard boundaries, out-of-order loading, and one/four-thread gathers, plus
32 changing CUDA graph replays across eight agents. It rejects heap-backed
sources and incomplete/overlapping shard coverage. Checkpoint export from this
backend is unsupported; the source checkpoint remains the portable artifact.
Its first full-model startup was stopped by the memory supervisor while a
parallel GPU kernel probe was running. This was test resource contention caused
by the experiment setup, not evidence of direct PLE's isolated memory demand.
The isolated successor stopped during PLE loading: its file-mapping validation
incorrectly assumed each tensor occupied one Linux VMA. Random-access advice
for one tensor can split the next tensor's mapping. This was reproduced on
checkpoint shard 1 without loading the model. The fix checks contiguous file
offsets, device/inode identity, and readable mappings across VMA boundaries.
All 128 real checkpoint shards then loaded, with exact-byte comparisons for
256 boundary rows, and the 32 changing CUDA graph replays passed again. A
same-file adjacent-tensor regression test also passed. The cheaper-host run
above subsequently passed the short full-model screen; long-context and broad
quality qualification remain outstanding.

`bench/tool_continuation.py` adds a tool round-trip check: distinct agents must
select their own record through a real parsed tool call, receive its result,
and return only their own verification code. It verifies server-reported input
usage against the requested context floor and requires a normal final stop.
Prompt preparation passed for eight 4k contexts and two 200k contexts; generated
tool calls and continuations remain to be tested against the running model.

## September 22 afternoon: parallel PLE reads (negative result)

The profiled MTP-2 trace showed one 21-34 ms GPU gap per step that ended with
the PLE staging copy, so the CPU gather was suspected. Cold-cache gathers of a
step's 384 rows took about 103 ms with the mapped four-thread path and about 5 ms
with 16-thread O_DIRECT pread in one idle-disk test; later idle-disk repeats were
noisy (26-35 ms buffered/direct). The pread backend (`FLASHNEXT_PLE_IO`,
buffered by default) is byte-identical to mapped reads on all 128 real shards.

Serving runs `pleio-buffered-mtp2` and `pleio-fp8dense-mtp2` are invalid tests
of these changes: the runtime's editable install imports the older
`/workspace/flashnext-gb10` checkout, which contains neither the pread backend
nor the dense-FP8 hook (the FP8 run's profile shows only BF16 dense GEMMs).
Their numbers restate the old configuration: 8/8 retrievals, stress
136.8-137.6 tok/s all-stream overlap, natural coding 92.6-103.2, mean accepted
length 2.1-2.2 on coding and 2.6 on retrieval. The launcher now puts the
current checkout first on PYTHONPATH.

A FULL_AND_PIECEWISE graph attempt failed at capture: the PLE prefetch uses an
eager CUDA-graph break that whole-step capture cannot contain.
