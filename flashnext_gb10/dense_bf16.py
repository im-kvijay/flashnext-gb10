"""Opt-in decode GEMM for the small BF16 projections (FLASHNEXT_DENSE_BF16=1).

Weights and activations stay BF16 (the checkpoint's values, FP32
accumulation); only the kernel changes. At decode batch sizes (M <= 32) the
per-layer hyperconnection, shared-expert, router and indexer projections are
2.6-6.6 MB each, and cuBLAS picks latency-bound kernels for them (the [1, 2560]
shared-expert gate takes about 43 us as a GEMV). A weight-streaming Triton
kernel with per-shape split-K reads them near memory bandwidth. Larger batches
(prefill) use the default GEMM.
"""
import os
import re

import torch
import triton
import triton.language as tl

TARGETS = re.compile(
    r'\.layers\.\d+\.((attn|mlp)_hyper_connection\.'
    r'(input_mix_weight_down_block_inject|input_mix_weight_down|input_mix_weight_up)'
    r'|mlp\.(gate|shared_expert_gate|shared_expert\.(gate_up_proj|down_proj))'
    r'|self_attn\.indexer\.index_qk_proj)$')
MTP_EXTRA = re.compile(r'^mtp\.(fc_embedding|fc_hidden|layers\.\d+\.self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj))$')

# (N, K) -> (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) at M <= 32 on GB10.
# From scripts/tune_dense_bf16.py; shapes not listed use a heuristic.
# FLASHNEXT_BF16_CONFIGS=<tuner --output JSON> adds or overrides entries (M=32 rows).
DECODE_CONFIGS = {}
# Shapes where cuBLAS was faster at M=32 on GB10 keep the default GEMM.
CUBLAS_SHAPES = set()


def _load_tuned(path):
    import json
    for key, row in json.loads(open(path).read()).items():
        shape, m = key.split('@')
        if m != 'M32':
            continue
        n, k = (int(v) for v in shape.split('x'))
        if row['kernel_us'] < row['cublas_us']:
            DECODE_CONFIGS[(n, k)] = tuple(row['config'])
        else:
            CUBLAS_SHAPES.add((n, k))


if os.environ.get('FLASHNEXT_BF16_CONFIGS'):
    _load_tuned(os.environ['FLASHNEXT_BF16_CONFIGS'])


@triton.jit
def _bf16_kernel(a_ptr, w_ptr, c_ptr, M, N, K, K_PER_SPLIT,
                 stride_am, stride_wn, stride_cm,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 ATOMIC: tl.constexpr, EVEN_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                        mask=offs_m[:, None] < M, other=0.0)
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=offs_n[:, None] < N, other=0.0)
        else:
            k_ok = offs_k < K
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                        mask=(offs_m[:, None] < M) & k_ok[None, :], other=0.0)
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=(offs_n[:, None] < N) & k_ok[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ATOMIC:
        tl.atomic_add(ptrs, acc, mask=mask, sem='relaxed')
    else:
        tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def default_config(N, K):
    """Enough CTAs to cover 48 SMs a few times while each streams >= 16 KB of weight."""
    block_n = 16 if N <= 64 else (32 if N <= 2048 else 64)
    block_k = 128 if K >= 128 else 64
    tiles = triton.cdiv(N, block_n)
    split_k = 1
    while tiles * split_k < 192 and K // (block_k * split_k * 2) >= 2 and K % (block_k * split_k * 2) == 0:
        split_k *= 2
    return block_n, block_k, split_k, 4, 4


def bf16_gemm(x: torch.Tensor, weight: torch.Tensor, config=None) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    block_n, block_k, split_k, warps, stages = config or DECODE_CONFIGS.get((N, K)) or default_config(N, K)
    block_m = 16 if M <= 16 else 32
    even_k = K % (block_k * split_k) == 0
    if not even_k:
        split_k = 1
        even_k = K % block_k == 0
    grid = (triton.cdiv(N, block_n), split_k, triton.cdiv(M, block_m))
    if split_k > 1:
        acc = torch.zeros((M, N), dtype=torch.float32, device=x.device)
        _bf16_kernel[grid](x, weight, acc, M, N, K, K // split_k, x.stride(0), weight.stride(0), acc.stride(0),
                           BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, ATOMIC=True, EVEN_K=even_k,
                           num_warps=warps, num_stages=stages)
        return acc.to(x.dtype)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    _bf16_kernel[grid](x, weight, out, M, N, K, K, x.stride(0), weight.stride(0), out.stride(0),
                       BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, ATOMIC=False, EVEN_K=even_k,
                       num_warps=warps, num_stages=stages)
    return out


def _decode_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 32 and tuple(weight.shape) not in CUBLAS_SHAPES:
        return bf16_gemm(x, weight)
    return torch.nn.functional.linear(x, weight)


def _decode_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def register_dense_bf16():
    from vllm.logger import init_logger
    from vllm.model_executor.layers import linear as linear_module
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op

    if getattr(LinearBase, '_flashnext_dense_bf16', False):
        return
    logger = init_logger('vllm.flashnext.dense_bf16')
    direct_register_custom_op(op_name='flashnext_bf16_decode_linear', op_func=_decode_linear,
                              fake_impl=_decode_linear_fake)

    class DecodeBF16LinearMethod(UnquantizedLinearMethod):
        def apply(self, layer, x, bias=None):
            if layer.weight.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
                return super().apply(layer, x, bias)
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if x2.stride(1) != 1:
                x2 = x2.contiguous()
            out = torch.ops.vllm.flashnext_bf16_decode_linear(x2, layer.weight)
            if bias is not None:
                out = out + bias
            return out.reshape(*shape[:-1], out.shape[-1])

    mtp = os.environ.get('FLASHNEXT_BF16_MTP', '1') == '1'
    original_init = LinearBase.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Most of these layers are built with quant_config=None, so the method is
        # chosen here; quantized or W8A16 layers keep theirs. Weights are
        # created after this by the subclass through the (inherited) method.
        prefix = getattr(self, 'prefix', '') or kwargs.get('prefix', '')
        if type(self.quant_method) is not UnquantizedLinearMethod:
            return
        tail = prefix[prefix.index('mtp'):] if prefix.startswith('mtp') or '.mtp.' in prefix else None
        if TARGETS.search(prefix) or (mtp and tail is not None and MTP_EXTRA.search(tail)):
            self.quant_method = DecodeBF16LinearMethod()
            logger.debug('FlashNext decode BF16 GEMM: %s', prefix)

    LinearBase.__init__ = __init__
    LinearBase._flashnext_dense_bf16 = True
    logger.info('FlashNext decode BF16 GEMM enabled for hyperconnection, router, shared-expert and indexer projections')
