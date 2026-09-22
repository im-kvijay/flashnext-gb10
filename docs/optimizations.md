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
given their scales. It needs the fidelity and suite comparison before use.

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
host-mapped input embeddings. Not yet demonstrated.

## Quality measurement

`bench/fidelity.py` replays 16 fixed sequences (eight recorded coding
continuations and eight real-source contexts; 32,768 scored positions) and
compares top-1 agreement, approximate KL and realized-token NLL between
configurations. Differences must be judged against a same-configuration
repeat, because batch-dependent kernels are not bitwise reproducible.
