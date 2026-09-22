# Experimental Flash Next DFlash attachment

This ports the DeepSpec attachment to the pinned vLLM V2 runner on one GB10.
It is not DFlash 2. The original checkpoint is
`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash` at
`9cd660f9050c92fedc88cbe547bd53af0392abe1`.

The reference source is DeepSpec revision
`54b35f57af721ffa8d55f1c0ae79f24952a82fd6`; its plugin and config are adapted
under the included MIT license. The vLLM patch is against
`a33b3bac5dad5654065adbdadc87f35118e482e3`, under vLLM's Apache-2.0 license.

`prepare.py` checks the five original source hashes, creates a separate package
overlay, applies the exact patch, and checks the resulting hashes. The baseline
package and downloaded draft config remain unchanged. Do not remove or upgrade
the baseline runtime while an overlay points to it.

```bash
$FLASHNEXT_RUNTIME/bin/python experiments/dflash/prepare.py \
  --output "$FLASHNEXT_DATA/dflash-adapter" \
  --draft "$FLASHNEXT_DATA/models/dflash-original"
export PYTHONPATH="$FLASHNEXT_DATA/dflash-adapter/overlay"
export FLASHNEXT_DFLASH_MODEL="$FLASHNEXT_DATA/dflash-adapter/draft"
$FLASHNEXT_RUNTIME/bin/python experiments/dflash/check_config.py \
  --model "$FLASHNEXT_MODEL" --draft "$FLASHNEXT_DFLASH_MODEL"
export FLASHNEXT_MTP=0 FLASHNEXT_DFLASH_K=4 FLASHNEXT_EAGER=1
unset FLASHNEXT_DRAFT_VOCAB
# Launch under scripts/supervise.py, as for the native baseline.
```

The attachment captures the learned HC contractions entering layers 4, 16, 24,
36, and 44, corresponding to the checkpoint's post-layer taps 3, 15, 23, 35,
and 43. It uses K query positions, samples from the anchor, and predicts the
next position. The V2 runner already implements that position kernel for
DSpark; this patch enables that layout for this DFlash checkpoint and accounts
for K-1 additional scheduler slots. Native MTP's absent hidden buffer is handled
explicitly. Default DFlash behavior is retained for other configurations.

The first experiment is TP1, eager, BF16 draft weights and BF16 draft KV, with
the target's embedding and full vocabulary head shared. K=4, 5, and 7 pass CPU
configuration and weight-header validation. `check_positions.py` passed eight
changing graph replays for each of K=4/5/7 and batch 1/8, using positions beyond
200k, nonidentity request-state mappings, prefill/bonus tokens and rejected KV
suffixes. The three pinned upstream input-preparation regressions also passed.
Full-model GPU inference, acceptance rates,
sampling equivalence, long-context capacity, and graph execution remain unproved.
The five full-attention draft layers add substantial KV memory at eight 200k
contexts; short-context speed does not establish the requested capacity.
