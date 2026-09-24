# Decode optimization log

Target: eight concurrent agents, at least 200k context each, at least 400
aggregate output tokens/s on natural workloads, with at least about 99% quality
retention. Baseline: `cheaper-direct-mtp2` (MTP-2, FP8 KV, direct PLE, piecewise
graphs): 134 tok/s on the all-eight fixed-output stress interval, 127 on short
retrieval, 93 on the profiled coding workload. See `roofline.md` for the
bandwidth ceiling.

## Step anatomy (profiled coding workload, about 172 ms per MTP-2 step)

| Component | ms/step | Notes |
|---|---|---|
| Routed-expert NVFP4 grouped GEMMs | 65 | gate/up 901 us, down 460 us per layer |
| BF16 dense GEMMs incl. lm_head | about 47 | GDN/QSA projections, HC, shared experts |
| GDN speculative state update | 17 | reads 1 state, writes k+1 states per layer |
| GPU idle | about 30 | one 21-34 ms gap per step, see below |
| MTP draft MoE (FP8) and other | about 13 | |

## PLE lookup stall (fixed, lossless)

Each step's GPU idle gap ends with the host-to-device copy of PLE rows: the
decoder waits for the CPU n-gram gather. Direct PLE copied rows out of the
checkpoint mapping, so every cold row was a page fault; only a few faults are
in flight at once (more OpenMP threads did not help through overlayfs). A
cold 384-row step (24 tokens x 16 heads, 160-byte rows) took 103 ms in
isolation. `MADV_WILLNEED` prefetching cut that to 26 ms; `pread` with 16-32
threads reached 5 ms on an idle disk. The loader's sequential reads reduce
random-read capacity to about 12k IOPS, so measurements taken during model
loading are not representative.

Correction: the run labelled `pleio-buffered-mtp2` (134.35 tok/s short
retrieval, 137.58 stress, 92.56 / 103.19 coding) imported an older installed
copy of this package, so only asynchronous ID staging was active and the
pread path was not exercised. The same happened to `pleio-fp8dense-mtp2`,
which is therefore a second baseline measurement (126.73 retrieval, 136.84
stress, 95.52 / 102.14 coding), not an FP8 result. Run-to-run noise is about
5%. `scripts/run_screen.py` now puts this tree first on `PYTHONPATH`, and the
server logs `FlashNext direct PLE lookup: io_mode=...` and the dense-FP8 hook
so that activation can be confirmed from `server.log`.

`FLASHNEXT_PLE_IO` selects `buffered` (default: pread with random advice,
page-cache hits stay cheap), `direct` (O_DIRECT) or `mapped` (previous path).
All three returned identical bytes over 64 real-checkpoint steps including
shard boundaries and invalid IDs, and over all 256 FP8 byte patterns.

## Full CUDA graphs

`FULL_AND_PIECEWISE` fails at capture: the CPU PLE gather needs a host
synchronization, which whole-step graph capture cannot contain.

`FLASHNEXT_PLE_PREFORWARD=1` removes that dependency: n-gram IDs depend only on
the step's input tokens, so they are computed and their rows gathered while
the runner prepares inputs, then copied into a persistent device buffer. The
forward only copies from that buffer, with no eager break, so decode can be
one FULL graph. `scripts/check_graph_replay.py --preforward [--direct]` passes
32 changing replays of a plain (unbroken) CUDA graph byte for byte. The cost is
that the gather no longer overlaps the embedding and first layer.

The asynchronous-ID path (`FLASHNEXT_ASYNC_PLE=1`) had a capture race: its
worker thread synchronized a CUDA event while the main thread was capturing
the next breakable segment, which invalidates a global-mode capture. Two
serving launches (`pleio2-mtp2`, `nospec-routing2`) failed at capture this way
before any measurement. During capture the ID copy is now awaited inside the
eager break and the worker does CPU-only work; serving replays are unchanged.

## Dense FP8 (numerics change; needs the fidelity check)

Decode-shaped timings on GB10 (M=24, CUTLASS block-FP8 including dynamic
activation quantization, versus BF16): GDN in 368 -> 199 us, GDN out
153 -> 64 us, QSA qkv 286 -> 165 us, QSA o 140 -> 64 us. FP8 is slower for
the small shared-expert matrices (about 60 us fixed cost), so they stay BF16.
Expected saving about 11-12 ms per step and about 2.5 GiB of weights.
`FLASHNEXT_DENSE_FP8=1` applies vLLM's online 128x128 block FP8 to those
projections at load time; the checkpoint is unchanged.
`scripts/derive_fp8_dense.py` writes an equivalent offline checkpoint.

Measured (`fp8dense2-mtp2`: FP8 dense, buffered pread PLE, asynchronous IDs,
MTP-2, piecewise graphs): 147.49 tok/s short retrieval, 151.64 on the
fixed-output stress interval, 110.64 / 114.02 on the natural coding workload
(draft acceptance 0.588 / 0.571). Against the two baseline measurements
(127-134, 134-138, 93-103) that is about 11-13%, combining FP8 dense and the
pread PLE path; `pleio3-mtp2` separates them.

Fidelity against the same-host baseline: on recorded coding continuations,
top-1 agreement 94.3%, approximate KL 0.023, NLL +0.006 nats/token. On raw
source-code contexts, top-1 agreement 81.3%, KL 0.34, NLL +0.069, with the
sign of the NLL change differing between sequences. Raw code inside a user
turn is high-entropy for this model (median 0.59 but 75th percentile 4.4
nats/token, frequent end-of-turn predictions), which amplifies any numeric
change; sparse-attention top-k selection also turns small differences into
discrete ones. The BF16-weight full-graph stack below (BF16 SSM state, MTP-3,
full graphs) scores 83.0% codebase top-1 against the same baseline, so the
codebase metric is dominated by run-to-run and state-precision noise; the
workload metric separates configurations better (95.1% versus FP8's 94.3%).

Rejected on the coding suite. LiveCodeBench v6 (30 problems, 16k token cap)
fell from 36.7% to 23.3%: four problems lost, none gained. All four were solved
by the baseline in 11-13k tokens and hit the length cap under FP8
(length-limited answers 18 -> 23 of 30), so FP8 dense makes reasoning longer
rather than visibly wrong. HumanEval (92.5%) and GSM8K (100%) were unchanged
and MMLU-Pro moved 78.6% -> 82.1% (within noise for 28 questions). This matches
the earlier observation that FP8 lm_head degraded Qwen3.8-27B. Weight-only FP8
(W8A16) is untested.

## Full-graph stack (lossless apart from BF16 SSM state)

`s2-full-bf16`: BF16 dense weights, MTP-3 with the MTP index shared across
draft iterations, BF16 Mamba/GDN SSM state, `FULL_AND_PIECEWISE` CUDA graphs
with pre-forward PLE, buffered pread PLE. 171.63 tok/s short retrieval, 160.51
fixed-output stress, 104.52 / 109.07 natural coding workload (acceptance
0.455, mean accepted length 2.37 versus 2.18 at MTP-2). That is about +28%
retrieval, +17% stress and +7% coding over the baseline. Coding gains least
because its lower acceptance makes the third draft position mostly wasted and
its more diverse n-grams make the now-serialized PLE gather colder. Fidelity
against the baseline: workload top-1 95.1%, KL 0.015, NLL +0.001 nats/token.

## Weight-only NVFP4 dense (numerics change)

`scripts/derive_nvfp4_dense.py` writes a sibling checkpoint whose 156 GDN and
QSA projections are W4A16_NVFP4 (FP8 per-16 scales chosen by a small error
search, one global scale per runtime-fused group), served by vLLM's FP4 Marlin
GEMM with BF16 activations. Dense bytes per step fall from 8.62 GB to about
4.8 GB and weights shrink by about 3.8 GB. Per-matrix relative weight error is
8.6% (FP8 block: about 2.6%), similar to the routed experts NVIDIA already
ships in NVFP4. The encoder reproduces the checkpoint's expert values exactly
given their scales.

Rejected on fidelity. `s5-nvfp4dense` (otherwise the full-graph stack):
170.10 tok/s short retrieval, 181.80 stress, 122.87 / 122.37 coding workload,
about +15% on coding over BF16 dense, and 7 GiB more host memory available.
Against the BF16 full-graph stack on the same host: workload top-1 89.0%,
KL 0.075, NLL +0.063 nats/token, all worse than FP8 dense, which already
failed the coding suite. Round-to-nearest NVFP4 is too coarse for these
projections; a calibrated quantizer (GPTQ-style error feedback) or a partial
target set would need to recover most of that gap before a suite run.

## Speculation depth

Coding-workload MTP-2 acceptance: 0.705 at position 1, 0.473 cumulative at
position 2 (mean accepted length 2.18). A third position would add about 0.3
tokens per step while verifying 33% more tokens, which the bandwidth model
makes roughly neutral. DFlash K=4 accepted 19.6% of drafts on the same
workload and is rejected.

## Capacity for 8 x 200k

The cache layout needs about 16.9 KB per token, about 28 GiB for eight
212,992-token requests. With a 6 GiB cache the host already fell to 15.1 GiB
available (17-24 GiB typical), so eight long contexts do not fit in the
current configuration. Candidate savings: dense FP8 (about 2.5 GiB), GDN
speculative state slots (about 1.8 GB), FP8 indexer keys, NVFP4 MTP experts,
host-mapped input embeddings.

First attempt (`cap200k-best`, full-graph BF16 stack, 24.5 GiB KV): the cache
held 1,652,167 tokens (7.76 x 212,992; 8 x 202,048 are needed), and host
available memory settled at about 9.5 GiB after graph capture (0.99 GiB of
graphs). About 100 s into the eight 200k prefills it fell to 7.2 GiB and the
supervisor's 8 GiB floor stopped the service before any decode. The likely
growth is prefill temporaries that scale with chunk size times context length
(sparse-attention indexer scores). The retry uses 1024-token prefill chunks
and expandable allocator segments. `FLASHNEXT_MIN_AVAILABLE_GIB` sets the
supervisor floor.

Without prefix priming the scheduler prefills one or two long prompts at a
time, so early agents finish before late ones start and eight-way decode is
never measured. That run (`cap200k-2`) did confirm the memory settings: host
available memory stayed near 8 GiB (minimum 7.5) through the 200k prefills.

Demonstrated (`cap200k-warm`, `profiles/gb10-8x200k.env`: full-graph BF16
stack, 24.5 GiB FP8 KV, 1024-token prefill chunks, expandable segments, 6 GiB
floor): eight distinct 200k-token codebase contexts primed through the prefix
cache (170-217 s each, about 1,000 prefill tokens/s), then eight simultaneous
2048-token continuations. 101.64 tok/s over the 148 s interval in which all
eight streams decoded (15,024 tokens), mean accepted length 2.51, no errors,
host memory above the floor throughout. At 4k context the same stack gives
104.5-109.1, so 200k context costs only a few percent of decode speed.

## FlashInfer b12x fused MoE (not usable at this memory budget)

`FLASHNEXT_MOE_BACKEND=flashinfer_b12x` JIT-compiles CuTe-DSL kernels during
warmup and on first requests of new token counts. Each compile needs several
GiB of host memory; three launches (`s4-b12x`, `s4c-b12x`, plus one that set
too small a KV cache for a 213k context) tripped the 8 GiB supervisor floor
before a measurement. It would need an offline kernel pre-build to evaluate.

## Quality measurement

`bench/fidelity.py` replays 16 fixed sequences (eight recorded coding
continuations and eight real-source contexts; 32,768 scored positions) and
compares top-1 agreement, approximate KL and realized-token NLL between
configurations. Differences must be judged against a same-configuration
repeat, because batch-dependent kernels are not bitwise reproducible.

## Fidelity noise floor and FP8 dense reinstated

`n1-old` repeats the baseline configuration with the original plugin copy;
`n2-new` runs the same configuration on the current tree. Against the baseline
reference:

| Run | Workload top-1 | Workload KL | Codebase top-1 | Codebase KL |
|---|---|---|---|---|
| Same configuration repeated (noise floor) | 95.2% | 0.0147 | 83.3% | 0.280 |
| Current tree, same configuration | 95.4% | 0.0146 | 82.6% | 0.285 |
| Full-graph BF16 stack (`s2-full-bf16`) | 95.1% | 0.0147 | 83.0% | 0.280 |
| FP8 dense, W8A8 (`fp8dense2-mtp2`) | 94.3% | 0.0230 | 81.3% | 0.343 |
| NVFP4 dense, round-to-nearest (`s5-nvfp4dense`, vs s2) | 89.0% | 0.0749 | 77.2% | 0.486 |

Greedy decoding with batch-dependent kernels is not reproducible, so most of the
differences previously attributed to configurations are run-to-run noise. The
current plugin and the full-graph stack are indistinguishable from the baseline.
FP8 dense adds a small real shift (KL +0.008); NVFP4 dense a large one.

The coding suite agrees: the BF16 full-graph stack (`suite-best`) scored exactly
what FP8 dense scored (LiveCodeBench 23.3%, 23 of 30 length-limited; HumanEval
92.5%; GSM8K 100%; MMLU-Pro 85.7% versus 82.1%). The single 36.7% baseline run
is the outlier, not an FP8 regression. FP8 dense is therefore reinstated.

## PLE lookup stall is storage-bound on the rented host

The 4k decode profile (`s6-best-prof`, 164 ms per MTP-3 step) shows 15 ms per
step of GPU idle between the n-gram ID copy and the PLE row upload; the 64k
profile (`p64k-best`) shows 35 ms. The host's overlay filesystem sustains about
12,000 random 160-byte reads per second at any thread count
(`iops.py`: 5.8k at 8 threads, 12.2k at 32, 11.7k at 128), so a step's cold rows
cost tens of milliseconds; with a larger KV cache the page cache holds fewer rows.
Only 38-43% of decoded tokens repeat a 3-gram already in their context (64-73%
for 2-grams), so a row cache cannot remove the misses. A DGX Spark's local NVMe
serves several hundred thousand random reads per second, where the same lookup
costs about 1-2 ms. Throughput measured here therefore understates a local
deployment. A GPU gather straight from the file mapping (GB10 reports pageable
memory access through host page tables) returns correct bytes but costs about
8 ms for 512 warm rows and 100-300 ms cold, so it is not used.

Other items in the 4k profile: routed-expert NVFP4 grouped GEMMs 76.6 ms per
step (1.06 + 0.53 ms per layer), dense BF16 GEMMs about 48 ms, GDN update 9.6 ms,
MTP draft MoE 5.4 ms. Isolated cuBLAS BF16 decode GEMMs already run at 200-222
GB/s (`experiments/megakernel/skinny_bf16.py`); a tuned Triton kernel saves only
2.1 ms per step.

## W8A16 dense (FP8 weights, BF16 activations)

`experiments/megakernel/w8a16.py` / `flashnext_gb10/dense_w8a16.py`: E4M3 weights
with one scale per 128x128 block (the FP8 dense rounding) streamed by a Triton
kernel with BF16 activations, so activation quantization is avoided. At M=32 it
reaches 195-229 GB/s: GDN input 409 -> 185 us, GDN output 150 -> 72 us, QSA qkv
307 -> 152 us, QSA output 144 -> 69 us, hyperconnection down 35 -> 17 us, shared
experts 32 -> 16 and 17 -> 9 us; 32.0 -> 15.1 ms per step for these
projections. `FLASHNEXT_DENSE_W8A16=1` covers GDN, QSA and shared experts;
`FLASHNEXT_W8A16_HC=1` adds hyperconnection projections and
`FLASHNEXT_W8A16_MTP=1` the draft block (draft proposals only).

## 8 x 200k with FP8 dense

`cap200k-fp8` (full-graph stack plus `FLASHNEXT_DENSE_FP8=1`, profile settings):
113.34 tok/s over 117 s with all eight 200k streams decoding, mean accepted
length 2.69, no errors, host available memory at least 10.4 GiB (BF16: 101.64,
2.51, about 7.5 GiB).

## Run-to-run divergence traced to FP4 activation quantization in the MoE

The "fidelity noise floor" above is not benign. `bench/determinism.py` scores
the same 2,304-token source file three times, alone, on one server. In every
configuration tried (production profile; eager without MTP, with BF16 KV,
FP32 GDN state and BF16 dense; plus PLE zeroed, state packing off, prefix
caching off) about 35% of positions move by more than 0.5 nats between
identical requests (max 11-18 nats), while the model's own greedy
continuations move at about 0.5% of positions. The same code span also scores
worse with more context (span at 512: 1.37 nats with the full 768-token
prefix, 0.62 with 128 tokens of context), and short-context scores of that
span range from 0.6 to 6.9 nats across runs.

Captures of every sub-module in layers 0-3 (`flashnext_gb10/module_capture.py`,
`experiments/module_diff.py`) of two identical requests:

| Module | Relative output difference (mean / max) |
|---|---|
| L0 GDN, L0 router | 0 (bitwise identical) |
| L0 routed MoE (identical input) | 6.8e-5 / 1.7e-3 |
| L1 routed MoE (input differs by 0.37%) | 4.6e-2 / 2.5e-1 |
| L3 routed MoE | 1.0e-1 / 3.4e-1 |

The GDN prefill kernels (FlashInfer and FLA) match an FP32 recurrence to
0.3% and are bitwise repeatable (`experiments/gdn_prefill_check.py`), and QSA
selections are complete below the budget (`experiments/qsa_check.py`). The
amplification is in the routed MoE and is not explained by routing: tokens
whose top-10 set is unchanged move by 4.5% on average. The served NVFP4 MoE
backend (`FLASHINFER_CUTLASS`) quantizes activations to FP4, so a 0.4%
input change flips FP4 rounding widely; across 40+ MoE layers this compounds
into the observed divergence.

`experiments/moe_reference.py` evaluates a layer's MoE in FP32 from the
checkpoint (dequantized NVFP4 weights, router, gated shared expert) on the
exact inputs vLLM saw, with and without an emulation of NVFP4 activation
quantization (static global scale, E4M3 scale per 16, E2M1 round-to-nearest):

| Layer | Served vs FP32 (W4A16) | W4A4 emulation vs FP32 | Served run-to-run | FP32 run-to-run | Input run-to-run |
|---|---|---|---|---|---|
| 1 | 8.1% | 7.5% | 4.6% | 0.52% | 0.37% |
| 3 | 10.7% | 9.4% | 10.0% | 3.6% | 2.3% |

The kernel is correct for W4A4; FP4 activations themselves inject 7-10%
error into every routed-MoE output, which the FP32 model does not amplify
but the served one does. NVFP4 weights with BF16 activations (Marlin, W4A16)
remove that error at unchanged decode speed (`m1-marlin`: workload 120/128,
prose 145 tok/s).

## Recommended profile without activation quantization

`profiles/gb10-8x200k.env` now uses Marlin W4A16 experts and W8A16 dense
projections. Measured on the rented GB10:

| Run | 8 x 200k | Workload 4k | Throughput 4k | Codebase repeat divergence (>0.5 nats) |
|---|---|---|---|---|
| FP8 dense, FP4-activation MoE (previous) | 113.3 | 112.0 / 122.6 | 175.7 | 35-38% |
| Marlin + W8A16, FP8 KV (`d10c`, `cap200k-marlin`) | 114.4 | 119.7 / 123.0 | 186.7 | 17-24% |
| Marlin + W8A16, BF16 KV (`d9`, 4k only) | - | 122.3 / 128.2 | 185.0 | 15-19% |

`cap200k-marlin`: eight distinct 200k codebase contexts, all eight decoding
together for 130 s, no errors, host available memory at least 10.6 GiB, mean
accepted length 2.2-2.6. The coding suite is unchanged within its small-sample
noise (HumanEval 92.5%, GSM8K 100%, LiveCodeBench 8/30 vs 7/30 with 22 of 30
length-limited, MMLU-Pro 22/28 vs 24/28).

With activation quantization removed, the served GDN layers, QSA attention
and routed MoE each match an FP32 transformers reference on captured inputs
to 0.5% (`experiments/gdn_reference.py`, `qsa_reference.py`,
`moe_reference.py`). The remaining run-to-run sensitivity on raw source code
and the observation that some code spans score worse with more context are
checked against a streaming full-model FP32 reference
(`experiments/model_reference.py`: NVFP4 weights dequantized, FP32 math,
dense attention over all 2048 tokens) on the served `codebase-0` sequence:

| Positions 256-2047 | Mean NLL |
|---|---|
| FP32 reference | 2.57 |
| Served, Marlin + BF16 KV (`d12`) | 2.47 |

| Comparison | Mean abs dlogp | Fraction > 0.5 nats |
|---|---|---|
| Served vs FP32 reference | 0.31 | 18% |
| Served vs served, identical requests | 0.35-0.38 | 17-24% |

The served model is as close to the FP32 reference as it is to itself, and
not worse on average. The reference shows the same context effect (span at
1536: 3.90 nats with full context, 3.05 with the shortened one). Both are
properties of the model on this input, not of the serving stack.

## FP8 KV cache and the MTP drafter

`experiments/drafter/diag_equivalence.py` replays captured prefill chunks
through a transformers copy of the MTP block and compares its LM-head hidden
state with vLLM's. With BF16 attention the match is poor (cosine 0.87-0.92,
draft argmax agreement 89-97%). Rounding the block's keys and values to FP8
E4M3 at unit scale, as the served FP8 KV cache does, reproduces vLLM
(cosine 0.993-0.998, agreement 98-99%). Any other FP8 rounding of the same
precision (per-tensor or per-head amax scales) lands back at cosine
0.86-0.91, so the block is highly sensitive to KV rounding: FP8 KV moves its
output by about 10% relative to BF16. On the target, BF16 KV scored 2.70
codebase NLL against 2.84 with FP8 KV, within the target's own run-to-run
spread. Keys have RMS 7 and max 92, values RMS 1.5 and max 15, so unit scale
is not the problem. FP8 KV stays in the 8 x 200k profile because BF16 KV for
eight 200k contexts does not fit next to the weights.

## PLE stall at 8 x 200k: row cache and early prefetch

`FLASHNEXT_PLE_TRACE` recorded every lookup of an 8 x 200k run with the trained
drafter (`cap200k-trained`, 114.9 tok/s): 509 rows per decode step and a
36.1 ms mean gather (p90 50.9 ms) while the GPU waits. The rented host's
container filesystem sits on a loop device in buffered mode
(`/var/lib/docker-loop.xfs`, `dio=0`), which serves about 12k random reads per
second at any thread count, buffered or O_DIRECT (`iops2.py`). About 45% of a
step's rows are n-grams never looked up before in the run, so no cache can
serve them (`experiments/ple_trace_stats.py`: a row cache of 1-6 GiB reaches
51-55% decode hits, a page cache of the same size 38-51%).

Two lossless changes, both opt-in:

- `FLASHNEXT_PLE_ROW_CACHE_GIB`: a two-way set-associative cache of 160-byte
  rows in front of `pread`, which also reads a row repeated within a step once.
- `FLASHNEXT_PLE_EARLY=1`: after sampling, each request's bonus token and its
  n-gram context are known, so its row IDs are computed as the next step will
  compute them and a helper thread reads the rows into the row cache while
  the drafter runs; the first draft token's rows follow after the first draft
  step. The step's own lookup is unchanged.

Same primed server (`cap200k-prof`, switched at run time via
`FLASHNEXT_PLE_CONTROL`), eight 200k contexts, 2048-token continuations:

| Lookup | tok/s | Accepted length |
|---|---|---|
| Page cache only, 32 readers | 115.6 | 2.65 |
| 3 GiB row cache, 64 readers | 121.6 | 2.63 |
| + early prefetch | 132.7 | 2.63 |

The first measurement after priming (same settings as the last row) was
125.5 tok/s; later sweep points may benefit from caches warmed by the earlier
ones. The 200k decode profile with early prefetch shows 93.6% GPU busy, about
9 ms of idle per step. On a host whose NVMe is not behind a loop device the
remaining reads take about 1-2 ms.

## Decode kernels at 8 x 200k (September 23-24)

Measured in isolation on GB10 (CUDA graphs, weights larger than L2):

| Component | Result |
|---|---|
| Routed MoE (Marlin W4A16) | 227 GB/s at the observed 129 distinct experts per layer for 32 tokens: at the bandwidth ceiling |
| GDN MTP decode, vLLM (state after every position written) | 309 us per layer |
| GDN replay decode (`FLASHNEXT_GDN_REPLAY=1`, one state write per step) | 167 us per layer, about 5 ms per step saved; matches vLLM's op to 2e-3 over multi-step runs with random acceptance |
| QSA decode indexer | 192 GB/s on BF16 keys; launch tuning saves 0.04 ms per step, not adopted |
| Small BF16 projections (hyperconnections, shared experts, router) | cuBLAS already 205-220 GB/s; a Triton decode GEMM adds 2-5% |
| Reduced draft head (29,977 rows) | cuBLAS FP32 GEMV at 187 GB/s; `FLASHNEXT_DRAFT_HEAD_W8=1` streams it as FP8 blocks |

The hyperconnection projections (about 1.9 GB per step) are bandwidth-bound in
BF16; `FLASHNEXT_W8A16_HC=1` now actually applies (they are built with
`quant_config=None`) and is gated on fidelity like the FP8 indexer keys
(`FLASHNEXT_INDEXER_KV_DTYPE=fp8`).

## Prefill

`bench/prefill.py` measures cold fills and file-sized appends at 200k depth on
real source. Baseline (1,024-token chunks, `prefill-prof`): cold 200k fill 217 s
(920 tok/s), 4k append at 200k depth 6.0 s, eight concurrent 12k appends 1,359
tok/s. Per 1,024-token step (about 780 ms): Marlin MoE 266 ms, which is the time
to stream all 68 GB of experts once (every chunk routes to all 512 experts);
W8A16 projections 138 ms, compute-bound; QSA 58 ms; about 165 ms of GPU idle,
mostly PLE row reads (24% of prefill wall time on the rented host's disk).

- `FLASHNEXT_PLE_PREFILL_AHEAD=8192` reads the next prompt rows on a CPU thread
  while the GPU computes: cold 200k 217 -> 152 s (1,314 tok/s), eight 12k
  appends 1,359 -> 2,040 tok/s. Lossless; in the 8 x 200k profile.
- W8A16 at prefill sizes: the Triton tile ran GDN in_proj and QSA qkv at 27
  TFLOPS; dequantizing the FP8 blocks to BF16 and calling cuBLAS reaches 55-88
  TFLOPS (BF16 cuBLAS: 89-99). `FLASHNEXT_W8A16_PREFILL=<tuner JSON>`.
- Larger chunks amortize the per-chunk expert streaming (4k and 8k measured
  next).
- 4,096-token chunks: cold 200k 152 -> 108 s; with the prefill W8A16 path 90 s
  (2,212 tok/s, 2.4x the baseline), eight 12k appends at 200k depth 3,221
  tok/s. 8,192-token chunks reached 2,160 tok/s at 64k but fell below the 6 GiB
  host floor during the 200k fill (6.29 GB available); 4,096 kept at least 7.1
  GB.
- Fidelity of the prefill path (`ab-prefillw8`) against the BF16-dense
  reference `m1-marlin`: workload top-1 96.6%, KL 0.0072; codebase top-1 86.7%,
  KL 0.202, NLL +0.015. The current profile without it scores 96.6% / 0.0072 and
  86.8% / 0.208 / +0.029, so the dequantize + cuBLAS path adds no measurable
  shift. Both are in `profiles/gb10-8x200k.env`.
- Decode A/B `ab-bf16` (Triton decode GEMM for the small BF16 projections):
  workload 121.3 / 128.3 tok/s vs 125.3 / 127.6 without, within noise; not
  adopted.
