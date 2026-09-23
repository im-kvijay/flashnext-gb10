"""W8A16 GEMM: FP8 E4M3 weights with 128x128 block scales, BF16 activations.

C[M, N] = A[M, K] @ dequant(W)[N, K]^T, FP32 accumulation. Same weight format
as vLLM's online per-block FP8 (weight [N, K] e4m3, weight_scale_inv
[ceil(N/128), K/128] fp32), but activations stay BF16, so the only numeric
change against BF16 is weight rounding. Decode (M <= 32) is weight-bandwidth
bound: FP8 halves the bytes of BF16.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, K_PER_SPLIT,
                  stride_am, stride_wn, stride_sn, stride_cm,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  SPLIT_K: tl.constexpr, ATOMIC: tl.constexpr):
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
        s = tl.load(s_ptr + (offs_n // 128) * stride_sn + (k_start + k0) // 128,
                    mask=offs_n < N, other=0.0)
        acc += tl.dot(a, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ATOMIC:
        tl.atomic_add(ptrs, acc, mask=mask, sem='relaxed')
    else:
        tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def w8a16_gemm(a, w, scale, block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=4, block_m=None):
    """a [M, K] bf16, w [N, K] float8_e4m3fn, scale [ceil(N/128), K/128] fp32 -> [M, N] bf16."""
    M, K = a.shape
    N = w.shape[0]
    # One scale per 128-wide K block: a K tile must not straddle two blocks.
    assert K % (block_k * split_k) == 0 and 128 % block_k == 0
    if block_m is None:
        block_m = 16 if M <= 16 else 32 if M <= 32 else 64
    grid = (triton.cdiv(N, block_n), split_k, triton.cdiv(M, block_m))
    if split_k > 1:
        acc = torch.zeros((M, N), dtype=torch.float32, device=a.device)
        _w8a16_kernel[grid](a, w, scale, acc, M, N, K, K // split_k, a.stride(0), w.stride(0), scale.stride(0),
                            acc.stride(0), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
                            ATOMIC=True, num_warps=num_warps, num_stages=num_stages)
        return acc.to(torch.bfloat16)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    _w8a16_kernel[grid](a, w, scale, out, M, N, K, K, a.stride(0), w.stride(0), scale.stride(0), out.stride(0),
                        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=1, ATOMIC=False,
                        num_warps=num_warps, num_stages=num_stages)
    return out


def quantize_block_fp8(w):
    """[N, K] bf16 -> (e4m3 [N, K], fp32 scale [ceil(N/128), K/128]) with amax/448 per 128x128 block."""
    N, K = w.shape
    pad = (-N) % 128
    wp = torch.nn.functional.pad(w.float(), (0, 0, 0, pad))
    blocks = wp.view(-1, 128, K // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(-1, K)[:N].contiguous(), scale.contiguous()


if __name__ == '__main__':
    import itertools, json, sys
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    from skinny_bf16 import SHAPES, bench
    torch.manual_seed(0)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_block_fp8_matmul
    except Exception:
        w8a8_block_fp8_matmul = None
    total = {'bf16': 0.0, 'w8a16': 0.0}
    for name, ((N, K), count) in SHAPES.items():
        if name == 'lm_head' or K % 128:
            continue
        nbytes = N * K
        copies = max(1, min(24, (256 << 20) // (2 * N * K)))
        ws = [torch.randn(N, K, device='cuda', dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        qs = [quantize_block_fp8(w) for w in ws]
        a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a, ws[0])
        t_bf16 = bench(lambda w: torch.nn.functional.linear(a, w), ws)
        best = None
        for bn, bk, sk, nw, ns in itertools.product((32, 64, 128), (64, 128), (1, 2, 4, 8), (4, 8), (3, 5)):
            if K % (bk * sk) or (N % 128 and bn > 64):
                continue
            programs = -(-N // bn) * sk
            if programs < 24 or programs > 8192:
                continue
            try:
                out = w8a16_gemm(a, qs[0][0], qs[0][1], bn, bk, sk, nw, ns)
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if err > 0.05:
                    continue
                t = bench(lambda q: w8a16_gemm(a, q[0], q[1], bn, bk, sk, nw, ns), qs)
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, dict(block_n=bn, block_k=bk, split_k=sk, num_warps=nw, num_stages=ns), err)
        t_w8a8 = None
        if w8a8_block_fp8_matmul is not None and N % 128 == 0:
            try:
                from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
                def w8a8(q):
                    aq, asc = per_token_group_quant_fp8(a, 128)
                    return w8a8_block_fp8_matmul(aq, q[0], asc, q[1], [128, 128], output_dtype=torch.bfloat16)
                t_w8a8 = bench(w8a8, qs)
            except Exception as e:
                t_w8a8 = f'error: {e}'[:80]
        row = dict(N=N, K=K, count=count, bf16_us=round(t_bf16 * 1e3, 1),
                   w8a16_us=round(best[0] * 1e3, 1) if best else None,
                   w8a16_gbs=round(nbytes / best[0] / 1e6, 1) if best else None,
                   w8a8_triton_us=round(t_w8a8 * 1e3, 1) if isinstance(t_w8a8, float) else t_w8a8,
                   rel_err_vs_bf16=round(best[2], 4) if best else None, config=best[1] if best else None)
        total['bf16'] += t_bf16 * count
        total['w8a16'] += (best[0] if best else t_bf16) * count
        print(json.dumps({name: row}), flush=True)
        del ws, qs
        torch.cuda.empty_cache()
    print(json.dumps({'step_ms': {k: round(v, 2) for k, v in total.items()}}))
