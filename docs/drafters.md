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

The checkpoint is downloaded for investigation. No local DFlash throughput or
quality result exists yet. Its serving adapter needs compatibility work against
the pinned runtime; do not enable it merely by changing a model name.

Native MTP is the first measured candidate. A draft-only reduced vocabulary
projection is also staged. It changes proposals, while leaving target logits,
target vocabulary, and target verification intact. Correct implementation and
sampling behavior still require validation before release.
