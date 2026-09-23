"""Opt-in W8A16 for BF16 target projections: FP8 weights, BF16 activations.

FLASHNEXT_DENSE_W8A16=1 quantizes the GDN and QSA projections and the shared
experts to E4M3 with one scale per 128x128 block after loading (the same
weight rounding as FLASHNEXT_DENSE_FP8), but activations stay BF16 and the
GEMM is a weight-streaming Triton kernel tuned for decode batches on GB10.
Routers, gates, the indexer, embeddings, lm_head and the MTP block stay BF16.
The checkpoint is unchanged.
"""
import os
import re

import torch
import triton
import triton.language as tl

TARGETS = re.compile(
    r'(?<!mtp)\.layers\.\d+\.(linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)'
    r'|self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj)'
    r'|mlp\.shared_expert\.(gate_up_proj|down_proj))$')
# Hyperconnection mixing projections: FLASHNEXT_W8A16_HC=1 (separately gated;
# their outputs feed residual-stream gates).
HC_TARGETS = re.compile(
    r'(?<!mtp)\.layers\.\d+\.(attn|mlp)_hyper_connection\.'
    r'(input_mix_weight_down_block_inject|input_mix_weight_down|input_mix_weight_up)$')
# MTP draft block: FLASHNEXT_W8A16_MTP=1. Draft proposals only; verified outputs are unchanged.
MTP_TARGETS = re.compile(
    r'^mtp\.(fc_embedding|fc_hidden|layers\.\d+\.(self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj)'
    r'|mlp\.shared_expert\.(gate_up_proj|down_proj)'
    r'|(attn|mlp)_hyper_connection\.(input_mix_weight_down_block_inject|input_mix_weight_down|input_mix_weight_up)))$')

# Measured on GB10 at M=32 (CUDA graph replay, weights larger than L2):
# (N, K) -> (BLOCK_N, SPLIT_K, num_warps, num_stages); BLOCK_K is 128.
DECODE_CONFIGS = {
    (16384, 2560): (32, 1, 8, 5),
    (2560, 6144): (64, 1, 4, 5),
    (13312, 2560): (32, 4, 4, 5),
    (1280, 2560): (32, 1, 8, 5),
    (2560, 640): (32, 1, 8, 3),
    (336, 10240): (64, 8, 4, 3),
    (320, 10240): (64, 8, 4, 3),
    (10240, 320): (64, 1, 4, 3),
}


@triton.jit
def _w8a16_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, K_PER_SPLIT,
                  stride_am, stride_wn, stride_sn, stride_cm,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SCALE_K: tl.constexpr, ATOMIC: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                    mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=offs_n[:, None] < N, other=0.0)
        s = tl.load(s_ptr + (offs_n // 128) * stride_sn + (k_start + k0) // SCALE_K,
                    mask=offs_n < N, other=0.0)
        acc += tl.dot(a, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ATOMIC:
        tl.atomic_add(ptrs, acc, mask=mask, sem='relaxed')
    else:
        tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def w8a16_gemm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    scale_k = K // scale.shape[1]
    if M <= 32:
        block_n, split_k, warps, stages = DECODE_CONFIGS.get((N, K), (64, 1, 4, 4))
        block_m = 16 if M <= 16 else 32
    else:
        # Prefill chunks are compute-bound; a square-ish tile is enough.
        block_n, split_k, warps, stages, block_m = 128, 1, 8, 3, 64
    block_k = min(128, scale_k)
    if K % (block_k * split_k):
        split_k = 1
    grid = (triton.cdiv(N, block_n), split_k, triton.cdiv(M, block_m))
    if split_k > 1:
        acc = torch.zeros((M, N), dtype=torch.float32, device=x.device)
        _w8a16_kernel[grid](x, weight, scale, acc, M, N, K, K // split_k, x.stride(0), weight.stride(0),
                            scale.stride(0), acc.stride(0), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                            SCALE_K=scale_k, ATOMIC=True, num_warps=warps, num_stages=stages)
        return acc.to(torch.bfloat16)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    _w8a16_kernel[grid](x, weight, scale, out, M, N, K, K, x.stride(0), weight.stride(0), scale.stride(0),
                        out.stride(0), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SCALE_K=scale_k, ATOMIC=False,
                        num_warps=warps, num_stages=stages)
    return out


def _w8a16_fake(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def quantize_block_fp8(weight: torch.Tensor):
    """[N, K] -> (e4m3 [N, K], fp32 [ceil(N/128), K/bk]); bk is 128, or 64 when K is not a multiple of 128.

    scale = block amax / 448 over 128 x bk blocks.
    """
    N, K = weight.shape
    bk = 128 if K % 128 == 0 else 64
    if K % bk:
        raise ValueError(f'W8A16 needs K divisible by 64, got {K}')
    pad = (-N) % 128
    padded = torch.nn.functional.pad(weight.float(), (0, 0, 0, pad))
    blocks = padded.view(-1, 128, K // bk, bk)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(-1, K)[:N].contiguous(), scale.contiguous()


def register_dense_w8a16():
    from vllm.logger import init_logger
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
    from vllm.utils.torch_utils import direct_register_custom_op

    if getattr(ModelOptMixedPrecisionConfig, '_flashnext_dense_w8a16', False):
        return
    logger = init_logger('vllm.flashnext.dense_w8a16')
    direct_register_custom_op(op_name='flashnext_w8a16_gemm', op_func=w8a16_gemm, fake_impl=_w8a16_fake)

    class W8A16LinearMethod(UnquantizedLinearMethod):
        def process_weights_after_loading(self, layer):
            if getattr(layer, '_flashnext_w8a16', False):
                return
            q, scale = quantize_block_fp8(layer.weight.data)
            layer.weight = torch.nn.Parameter(q, requires_grad=False)
            layer.register_buffer('flashnext_weight_scale', scale, persistent=False)
            layer._flashnext_w8a16 = True

        def apply(self, layer, x, bias=None):
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if x2.stride(1) != 1:
                x2 = x2.contiguous()
            out = torch.ops.vllm.flashnext_w8a16_gemm(x2, layer.weight, layer.flashnext_weight_scale)
            if bias is not None:
                out = out + bias
            return out.reshape(*shape[:-1], out.shape[-1])

    original = ModelOptMixedPrecisionConfig.get_quant_method
    hc = os.environ.get('FLASHNEXT_W8A16_HC') == '1'
    mtp = os.environ.get('FLASHNEXT_W8A16_MTP') == '1'

    def get_quant_method(self, layer, prefix):
        is_mtp = prefix.startswith('mtp') or '.mtp.' in prefix
        if isinstance(layer, LinearBase) and (
                (not is_mtp and (TARGETS.search(prefix) or (hc and HC_TARGETS.search(prefix))))
                or (is_mtp and mtp and MTP_TARGETS.search(prefix[prefix.index('mtp'):]))):
            logger.info('FlashNext dense W8A16: %s', prefix)
            return W8A16LinearMethod()
        return original(self, layer, prefix)

    ModelOptMixedPrecisionConfig.get_quant_method = get_quant_method
    ModelOptMixedPrecisionConfig._flashnext_dense_w8a16 = True
