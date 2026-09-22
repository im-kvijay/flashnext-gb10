"""Opt-in online block-FP8 for the large BF16 target projections.

The pinned checkpoint keeps GDN and QSA projections in BF16. With
FLASHNEXT_DENSE_FP8=1 these layers load their BF16 weights and quantize them to
E4M3 with one scale per 128x128 block (vLLM's online per-block method). The
checkpoint is unchanged. Shared experts are excluded because FP8 is slower than
BF16 at their size; routers, GDN gates, the indexer, embeddings, lm_head and
the MTP draft block stay BF16. This changes numerics and needs a paired
quality comparison.
"""
import re

TARGETS = re.compile(
    r'(?<!mtp)\.layers\.\d+\.(linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)'
    r'|self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj))$')


def register_dense_fp8():
    from vllm.logger import init_logger
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod

    if getattr(ModelOptMixedPrecisionConfig, '_flashnext_dense_fp8', False):
        return
    logger = init_logger('vllm.flashnext.dense_fp8')
    original = ModelOptMixedPrecisionConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        if (isinstance(layer, LinearBase) and not prefix.startswith('mtp')
                and '.mtp.' not in prefix and TARGETS.search(prefix)):
            logger.info('FlashNext dense FP8: %s', prefix)
            return Fp8PerBlockOnlineLinearMethod()
        return original(self, layer, prefix)

    ModelOptMixedPrecisionConfig.get_quant_method = get_quant_method
    ModelOptMixedPrecisionConfig._flashnext_dense_fp8 = True
