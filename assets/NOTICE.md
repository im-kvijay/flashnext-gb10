# Notice for files in assets/

`drafter/mtp_trained.pt.part*` together form a retrained multi-token-prediction
(MTP) drafter derived from NVIDIA's
[Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
checkpoint. The original is licensed by NVIDIA Corporation under the
NVIDIA Open Model License; this derivative is distributed under the same
license, and its use is subject to that license's terms. The drafter only
proposes tokens for speculative decoding. The served model's outputs come from
NVIDIA's unmodified checkpoint weights.

`reference/` holds measurement records (model scores, benchmark items and
outputs) from the reference GB10, used by `scripts/verify_fidelity.sh` and
`scripts/quality_check.sh`. Benchmark items remain under their datasets' own
licenses (LiveCodeBench, HumanEval, GSM8K, MMLU-Pro).
