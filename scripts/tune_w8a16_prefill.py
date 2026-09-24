"""Tune the W8A16 GEMM (flashnext_gb10/dense_w8a16.py) at prefill chunk sizes.

Prefill chunks are compute-bound: at 1,024 tokens the W8A16 projections ran
at about 64 TFLOPS on GB10. For each served shape and chunk size this times
the current prefill tile, a grid of Triton tiles, and dequantizing the FP8
blocks to BF16 followed by cuBLAS (the dequantized weight is exactly what the
Triton kernel multiplies by, up to where the scale is applied), with plain
BF16 cuBLAS as the ceiling. Prints the best choice per shape.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashnext_gb10.dense_w8a16 import _w8a16_kernel, dequantize_block_fp8, quantize_block_fp8  # noqa: E402

import triton  # noqa: E402

SHAPES = [(16384, 2560), (2560, 6144), (13312, 2560)]  # GDN in_proj_qkvz, GDN out_proj / QSA o_proj, QSA qkv_proj


def timed(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def triton_gemm(x, q, scale, block_m, block_n, warps, stages):
    M, K = x.shape
    N = q.shape[0]
    scale_k = K // scale.shape[1]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(N, block_n), 1, triton.cdiv(M, block_m))
    _w8a16_kernel[grid](x, q, scale, out, M, N, K, K, x.stride(0), q.stride(0), scale.stride(0), out.stride(0),
                        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=min(128, scale_k), SCALE_K=scale_k, ATOMIC=False,
                        num_warps=warps, num_stages=stages)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, nargs='+', default=[1024, 4096, 8192])
    p.add_argument('--output')
    a = p.parse_args()
    torch.manual_seed(0)
    report = {}
    for N, K in SHAPES:
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16) * 0.02
        q, scale = quantize_block_fp8(w)
        deq = dequantize_block_fp8(q, scale)
        for M in a.m:
            x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
            flops = 2 * M * N * K
            ref = x.float() @ deq.float().T
            rows = {}
            rows['bf16_cublas'] = timed(lambda: torch.nn.functional.linear(x, w))
            rows['dequant_cublas'] = timed(lambda: torch.nn.functional.linear(x, dequantize_block_fp8(q, scale)))
            rows['current'] = timed(lambda: triton_gemm(x, q, scale, 64, 128, 8, 3))
            best = ('current', (64, 128, 8, 3))
            for bm, bn, warps, stages in itertools.product((64, 128), (64, 128, 256), (4, 8), (2, 3, 4)):
                cfg = (bm, bn, warps, stages)
                try:
                    out = triton_gemm(x, q, scale, *cfg)
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    if err > 1e-2:
                        continue
                    t = timed(lambda c=cfg: triton_gemm(x, q, scale, *c))
                except Exception:
                    continue
                rows[str(cfg)] = t
                if t < rows[best[0]]:
                    best = (str(cfg), cfg)
            choice = min(('dequant_cublas', best[0]), key=lambda k: rows[k])
            report[f'{N}x{K}@M{M}'] = dict(us=rows, best=choice, triton_best=best[1])
            print(f'N={N:5d} K={K:5d} M={M:5d}: bf16 cuBLAS {flops / rows["bf16_cublas"] / 1e6:6.1f} TFLOPS, '
                  f'current {flops / rows["current"] / 1e6:6.1f}, best triton {best[1]} '
                  f'{flops / rows[best[0]] / 1e6:6.1f}, dequant+cuBLAS {flops / rows["dequant_cublas"] / 1e6:6.1f} '
                  f'-> {choice}', flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
