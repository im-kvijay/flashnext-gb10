# Single-GB10 bandwidth ceiling

Measured September 22 on the rented GB10 (idle GPU, 2.41 GHz SM clock), with
`scripts/probe_bandwidth.py`: 240-243 GB/s for a BF16 read reduction, 224 GB/s
for a copy, and 235 GB/s for BF16 GEMMs shaped like decode projections (8 or 24
activation rows). The GB10 specification is 273 GB/s.

Decode at eight concurrent agents is memory-bound. Each target step must read
every BF16 dense weight once, each distinct routed NVFP4 expert used by that
step's tokens once, and the FP32 GDN recurrent state in both directions. Each
MTP draft step reads the MTP block and the draft projection. `scripts/roofline.py`
computes this from the checkpoint's safetensors headers:

| Per-step component (pinned NVIDIA checkpoint) | Bytes |
|---|---|
| BF16 dense: GDN, QSA attention, hyperconnections, shared experts, router, lm_head | 8.62 GB |
| One routed NVFP4 expert (weights and scales) | 2.76 MB |
| FP32 GDN state per agent (read or write) | 113 MB |
| MTP block per draft step, with 29,977-row draft projection | about 0.33 GB + experts |

Independent uniform routing gives an upper bound on distinct experts. The
profiled natural coding run at MTP-2 spent 1.36 ms per layer in the two routed
grouped GEMMs; at 235 GB/s that corresponds to about 116 distinct experts for 24
tokens, 0.6 of the uniform estimate. Floors at 240 GB/s with that calibration
(`--expert-fraction 0.6 --draft-vocab 29977`):

| Draft tokens k | GB/step | Step floor | Ceiling if every draft accepted | Mean accepted length needed for 400 tok/s |
|---|---|---|---|---|
| 0 | 16.4 | 68 ms | 117 tok/s | 3.41 (max 1) |
| 2 | 27.2 | 113 ms | 212 tok/s | 5.67 (max 3) |
| 3 | 31.6 | 132 ms | 243 tok/s | 6.59 (max 4) |
| 5 | 38.9 | 162 ms | 296 tok/s | 8.10 (max 6) |
| 7 | 44.6 | 186 ms | 345 tok/s | 9.29 (max 8) |

Even at an optimistic 0.4 expert fraction, 400 tok/s needs k=6 with 6.84 of 7
drafts accepted on average. Measured MTP-2 mean accepted length in the
`cheaper-direct-mtp2` run was 2.18 on the natural coding workload (58.9% of
drafts), 2.49 on the reasoning retention screen and 2.65 on short retrieval. With experts treated as free, BF16 dense weights, GDN
state and draft reads alone need 49 ms per k=2 step.

These floors omit long-context indexer and KV reads, speculative rollback state,
activations, launch gaps and kernel inefficiency, so real throughput is lower.
The observed MTP-2 run takes about 172 ms per step against a 113 ms floor: GPU
idle time was 19% of the profiled interval, and kernels ran below peak bandwidth.

The expert fraction is inferred, not counted: it assumes the NVFP4 grouped GEMM
runs near 235 GB/s. If that kernel is tile-bound at two to three tokens per
expert, fewer experts are read, the fraction is nearer 0.4, and MoE kernel
efficiency becomes the largest lossless lever. Counting distinct routed experts
per layer on the coding workload settles this.

Conclusion: with the pinned checkpoint and no change in model outputs, 400
aggregate output tokens/s at eight concurrent agents is above one GB10's memory
bandwidth. At the measured coding acceptance of 2.18, the k=2 floor gives about
155 tok/s at fraction 0.6 and 190 at 0.4, before long-context reads and launch
gaps. Reaching 400 requires fewer bytes per step, which changes the weights, or
more memory bandwidth. Two linked GB10s halve per-device bytes but add two
collectives per layer; at realistic acceptance that estimate is roughly 300 to
350 tok/s and has not been measured.
Lossless entropy coding of BF16 dense weights can remove only part of the dense
component.
