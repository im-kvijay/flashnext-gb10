# Validation against the base model

"Base" is NVIDIA's Qwen3.8-Flash-Next-NVFP4 checkpoint served with stock vLLM
settings at the pinned commit: default NVFP4 MoE kernel (which also rounds
activations to FP4), BF16 dense layers and KV cache, FP32 GDN state, the
checkpoint's MTP drafter with its full vocabulary (`base-quality`). "Recipe"
is `profiles/gb10-8x200k.env` with the retrained drafter, installed from the
release bundle with `scripts/setup_gb10.sh` and started with
`scripts/start.sh` on the reference GB10 (September 24).

| Check | Base | Recipe |
|---|---|---|
| Everyday tasks (`bench/normal_tasks.py`, 8 tasks x 3) | 24/24 | 24/24 |
| Multi-turn tool use, 8 agents at 4k | 8/8 | 8/8 |
| Long-context retrieval (`bench/retention.py`) | 10/12 | 12/12 |
| HumanEval (40) | 37 | 37 (0 flips) |
| GSM8K (20) | 20 | 20 |
| MMLU-Pro (28) | 22 | 21 (2 lost, 1 gained) |
| LiveCodeBench v6 (30), 16k-token budget | 12 | 9 (5 lost, 2 gained) |
| Fidelity vs unquantized dense, coding-agent text: top-1 / KL | 94.6% / 0.022 | 96.6% / 0.007 |
| Fidelity vs unquantized dense, raw source: top-1 / KL | 81.7% / 0.32 | 86.5% / 0.21 |

The teacher-forced fidelity measurement, the most sensitive of these, puts
the recipe closer to the unquantized model than stock serving: stock vLLM's
NVFP4 MoE kernel quantizes activations to FP4, the recipe keeps them in BF16.

LiveCodeBench at a 16k-token budget does not separate configurations. In
every one of six suite runs (base, recipe and four earlier configurations)
all LiveCodeBench failures but one were answers cut off at the budget; every
item both runs finished was correct in both. Scores ranged from 7 to 12 of 30
and configurations differed by 2 to 7 items; the base differs from every other
configuration by 6 or 7. All five of the recipe's lost items were cut off at
16,384 tokens (the base finished them in 9,400-14,000). HumanEval (37/40) and
GSM8K (20/20) are identical in all six runs; MMLU-Pro ranges 21-24. A
32k-token LiveCodeBench comparison, where most answers can finish, is below.

The shipped profile uses 2,048-token prefill chunks (the table above was
measured with 4,096). Its 8 x 200k run (`final2-8x200k`) re-checked fidelity
(96.5% / 0.0071 coding-agent text, 85.8% / 0.219 raw source) and everyday
tasks (24/24) on the same server.
