# Drafter investigation

Checked September 22, 2026. No Flash Next-specific DFlash 2 checkpoint was found
in the Hugging Face model catalog searches for `Flash-Next-DFlash`,
`Qwen3.8-Flash-Next-DFlash2`, and `Qwen3.8-Flash-Next-DFlash-2`.
This is a search result, not a claim that none can exist elsewhere.

The official [DFlash 2 checkpoint](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
targets the dense Qwen3.8-27B. Its architecture and conditioning do not make it a
drop-in drafter for Flash Next. Its H200 results cannot be used as GB10 estimates.

The compatible candidate is
[PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash),
revision `9cd660f9050c92fedc88cbe547bd53af0392abe1`. This is the original DeepSpec
block drafter, not DFlash 2. It has 498,106,880 BF16 parameters and shares the
target embedding and output head. It conditions on five HC-contracted target
states at layers 3, 15, 23, 35, and 43. Those exact taps matter for compatibility.

The authors report 3.87% aggregate improvement over tuned native MTP at block 5,
using two GB10s, concurrency one, 8k context, and thinking disabled. Code was
roughly tied and chat slower. Their graph integration is not working. These
conditions do not qualify our single-GB10, eight-agent, 200k, xhigh workload.

The checkpoint is downloaded and its LFS hash verified. An isolated
[V2 adapter](../experiments/dflash/README.md) ports the HC taps and anchor layout
to the pinned runtime. All 58 checkpoint tensors match the required BF16 shapes;
K=4, 5, and 7 pass the CPU configuration check. Do not enable it merely by
changing a model name.

The first full-model DFlash run (`dflash-k4-eager`, K=4, eager, BF16 draft,
FP8 target KV, direct and asynchronous PLE) passed 8/8 short retrievals with
natural stops. Its all-eight overlap rate was 87.45 tokens/s on retrieval and
105.99 on the fixed-output stress test. On the natural coding workload it reached
57.82 tokens/s during overlap with 9,438 of 48,124 drafts accepted (19.6%, about
1.8 tokens per step). The graph-enabled native MTP-2 run reached 93.02 on its
coding workload while profiled. Eager execution penalizes DFlash, but its
acceptance is too low to overcome that with graphs. DFlash is rejected for this
workload; the run was stopped before retention and tool screens.

Native MTP is the first measured candidate. A draft-only reduced vocabulary
projection is also staged. It changes proposals, while leaving target logits,
target vocabulary, and target verification intact. Correct implementation and
sampling behavior still require validation before release.

## Retrained MTP drafter (self-distillation)

The served drafter is the checkpoint's own MTP block with its non-expert
weights retrained on the served target (`experiments/drafter/`,
`scripts/build_drafter.sh`). The target generates answers to
KodCode-Light-RL-10K prompts (disjoint from every benchmark here); an eager
server replays them and saves the drafter's inputs per prefill chunk
(`FLASHNEXT_CAPTURE_MTP_DIR`); `train_mtp.py` unrolls the three draft steps
exactly as served (FP8 KV rounding, reused step-1 keys and values, reduced
draft vocabulary) and minimizes KL to the target's next-token distribution.
Routed experts, embeddings and lm_head stay frozen unless `--train-experts`.
The result is loaded with `FLASHNEXT_MTP_OVERRIDE=<file.pt>`; the target and
its verification are unchanged, so outputs are unchanged.

| Drafter | Data | Held-out expected accepted length | Served accepted length, 8 x 200k |
|---|---|---|---|
| checkpoint MTP | - | 2.65 | 2.53 (`cap200k-marlin`) |
| retrained, 1 epoch | 96 generations | 2.87 | 2.64 (`cap200k-trained`) |
