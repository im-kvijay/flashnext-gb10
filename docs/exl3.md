# EXL3 comparison status

The read-only delegated investigation found a plausible memory-saving candidate,
not evidence that EXL3 reaches 400 aggregate output tokens/s with eight 200k
agents. No EXL3 full-model benchmark has run in this project.

Candidate source: [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3/tree/329e051385505b6ba981138d86a90bffe032c831)
at `329e051385505b6ba981138d86a90bffe032c831`. This ARM64/GB10 fork is distinct
from upstream `turboderp-org/exllamav3` at
`6b84a21b6f1e5da3f291b9e1019061f0de788279`. CPU-expert and tensor-parallel
stubs in the ARM build are not working implementations of those features.

First candidate checkpoint: `turboderp/Qwen3.8-Flash-Next-exl3`, revision
`55a732e0c4c3d4614bc42b68493bb930d9b02c0a` (`4.05bpw_h6_ng6`). Listed tensors
occupy about 99.94 GiB, including 36.36 GiB of PLE and 63.58 GiB of other weights.
The 3.05-bit variant uses about 79.15 GiB total. These are different quantizations
from the pinned NVIDIA checkpoint, including different PLE representations;
neither is an exact-byte substitute for the existing FP8 PLE offload.

Source-derived sizing for eight requests with 200,000 input and 16,384 output
tokens is about 29.54 GiB for Q8 KV plus the full-precision indexer fields.
FP16 KV is about 49.67 GiB. Recurrent states, rollback/checkpoint buffers,
workspaces, graphs, the process, and the operating system require more memory.
The 4.05-bit/Q8/MTP-3 configuration is approximately 100.5 GiB before remaining
runtime overhead; this is a sizing estimate, not demonstrated capacity.

The fork defaults to `EXL3_GR_INT8=1`, which changes target hyperconnection
precision, and `EXL3_MTP_HEAD_N=65536`, which prunes the draft vocabulary. Set
both to `0` for the first fidelity comparison. Its MTP source calls the stream
tap a semantic guess, so acceptance and verification need direct testing.

The [published GB10 recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/blob/e0ebee8ddc4391b5e66d86fdce993b0756e00986/README.md)
reports about 152 aggregate tokens/s for eight short streams and 135.8 for
eight 3k streams with forced 128-token output. Its 240k result concerns one
stream. Those measurements do not qualify this project's concurrency, natural
workload, quality, or throughput requirements.

The bounded comparison should use native EXL3 plus TabbyAPI, an aggregate cache
pool of at least 1,732,608 tokens and eight recurrent slots, exact checkpoint
revisions, disk PLE, and the same memory supervisor and prompts as the NVIDIA
baseline. Test speculation off, MTP-1, then MTP-3; require short correctness and
tool parsing before 1/2/4/8 distinct long requests. Record normal EOS, accepted
output tokens, prefill latency, preemptions, memory, and failures. Any promotion
also needs paired retention evidence against the NVIDIA reference.
