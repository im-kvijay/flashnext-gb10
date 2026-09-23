"""Weight-streaming BF16 GEMM for decode batches (M <= 32) on GB10.

C[M, N] = A[M, K] @ W[N, K]^T with FP32 accumulation. Each program owns a
BLOCK_N slice of W and a K range (split-K); partial sums are added with FP32
atomics into a zeroed workspace and cast once. Decode GEMMs read each weight
byte once, so the goal is sustained DRAM bandwidth, not FLOPs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_kernel(a_ptr, w_ptr, acc_ptr, M, N, K, K_PER_SPLIT,
                   stride_am, stride_wn, stride_accm,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(w))
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = acc_ptr + offs_m[:, None] * stride_accm + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(ptrs, acc, mask=mask)
    else:
        tl.atomic_add(ptrs, acc, mask=mask, sem='relaxed')


def skinny_gemm(a, w, block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=4, out=None, acc=None):
    M, K = a.shape
    N = w.shape[0]
    assert M <= 32 and K % (block_k * split_k) == 0
    if acc is None:
        acc = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    elif split_k > 1:
        acc.zero_()
    grid = (triton.cdiv(N, block_n), split_k)
    _skinny_kernel[grid](a, w, acc, M, N, K, K // split_k, a.stride(0), w.stride(0), acc.stride(0),
                         BLOCK_M=32 if M > 16 else 16, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
                         num_warps=num_warps, num_stages=num_stages)
    return acc.to(torch.bfloat16) if out is None else out.copy_(acc)


SHAPES = {  # (N, K): count per decode step
    'gdn_in_qkvz': ((16384, 2560), 36),
    'gdn_out': ((2560, 6144), 36),
    'qsa_qkv': ((13312, 2560), 12),
    'qsa_o': ((2560, 6144), 12),
    'hc_down': ((320, 10240), 96),
    'hc_up': ((10240, 320), 96),
    'shared_gate_up': ((1280, 2560), 48),
    'shared_down': ((2560, 640), 48),
    'router': ((512, 2560), 48),
    'lm_head': ((248320, 2560), 1),
}


def bench(fn, weights, iters=3):
    """Time one call per weight copy inside a CUDA graph (copies defeat the L2)."""
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for w in weights:
            fn(w)  # warm-up / compile outside capture
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for w in weights:
                fn(w)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay(); torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / len(weights)


if __name__ == '__main__':
    import itertools, json, sys
    torch.manual_seed(0)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    results = {}
    for name, ((N, K), count) in SHAPES.items():
        nbytes = N * K * 2
        copies = max(1, min(24, (256 << 20) // nbytes))
        weights = [torch.randn(N, K, device='cuda', dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a, weights[0])
        t_cublas = bench(lambda w: torch.nn.functional.linear(a, w), weights)
        best = None
        for bn, bk, sk, nw, ns in itertools.product((64, 128), (128, 256), (1, 2, 4, 8), (4, 8), (3, 5)):
            if K % (bk * sk) or (sk > 1 and K // sk < bk):
                continue
            programs = -(-N // bn) * sk
            if programs < 24 or programs > 8192:
                continue
            acc = torch.zeros((M, N), dtype=torch.float32, device='cuda')
            try:
                out = skinny_gemm(a, weights[0], bn, bk, sk, nw, ns)
                err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
                if err > 2e-2:
                    continue
                t = bench(lambda w: skinny_gemm(a, w, bn, bk, sk, nw, ns, acc=acc), weights)
            except Exception as e:  # resource limits for some configs
                continue
            if best is None or t < best[0]:
                best = (t, dict(block_n=bn, block_k=bk, split_k=sk, num_warps=nw, num_stages=ns), err)
        gbs = lambda t: nbytes / t / 1e6
        results[name] = dict(N=N, K=K, count=count, cublas_us=round(t_cublas * 1e3, 1), cublas_gbs=round(gbs(t_cublas), 1),
                             triton_us=round(best[0] * 1e3, 1) if best else None,
                             triton_gbs=round(gbs(best[0]), 1) if best else None,
                             config=best[1] if best else None, max_rel_err=best[2] if best else None,
                             step_saving_ms=round((t_cublas - best[0]) * count, 2) if best else None)
        print(json.dumps({name: results[name]}), flush=True)
        del weights
        torch.cuda.empty_cache()
    print('total step saving ms', round(sum(r['step_saving_ms'] or 0 for r in results.values() if (r['step_saving_ms'] or 0) > 0), 2))
