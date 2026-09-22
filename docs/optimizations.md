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

`FLASHNEXT_PLE_IO` selects `buffered` (default: pread with random advice,
page-cache hits stay cheap), `direct` (O_DIRECT) or `mapped` (previous path).
All three returned identical bytes over 64 real-checkpoint steps including
shard boundaries and invalid IDs, and over all 256 FP8 byte patterns.

## Full CUDA graphs

`FULL_AND_PIECEWISE` fails at capture: the CPU PLE gather needs a host
synchronization, which whole-step graph capture cannot contain. Piecewise
graphs with an eager break at the PLE layer remain required while the table
is gathered on the CPU.

## Dense FP8 (numerics change; needs the fidelity check)

Decode-shaped timings on GB10 (M=24, CUTLASS block-FP8 including dynamic
activation quantization, versus BF16): GDN in 368 -> 199 us, GDN out
153 -> 64 us, QSA qkv 286 -> 165 us, QSA o 140 -> 64 us. FP8 is slower for
the small shared-expert matrices (about 60 us fixed cost), so they stay BF16.
Expected saving about 11-12 ms per step and about 2.5 GiB of weights.
`FLASHNEXT_DENSE_FP8=1` applies vLLM's online 128x128 block FP8 to those
projections at load time; the checkpoint is unchanged.
`scripts/derive_fp8_dense.py` writes an equivalent offline checkpoint.

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
