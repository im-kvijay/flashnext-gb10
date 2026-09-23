"""Tune and check the decode BF16 GEMM (flashnext_gb10/dense_bf16.py) on the served shapes.

For each (N, K) and decode batch M, compares against an FP32 reference, then
times cuBLAS and candidate kernel configs inside CUDA graphs over rotating
weight copies (larger than L2, as in a decode step). Prints the best config
per shape as a DECODE_CONFIGS entry. Needs the GPU to itself.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashnext_gb10.dense_bf16 import bf16_gemm, default_config  # noqa: E402

SHAPES = [
    (336, 10240), (320, 10240), (10240, 320),    # hyperconnection down (+inject) / up
    (1280, 2560), (2560, 640),                   # shared expert gate_up / down
    (512, 2560), (1, 2560),                      # router, shared-expert gate
    (640, 2560),                                 # QSA indexer q/k projection
    (12288, 2560), (2560, 6144), (2560, 2560),   # MTP q / o / fc
]


def graph_time(fn, copies, iters=20):
    """Mean microseconds per call over a CUDA graph cycling through weight copies."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for w in copies:
            fn(w)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in copies:
            fn(w)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (iters * len(copies))


def candidates(N, K):
    for block_n, block_k, split_k, warps, stages in itertools.product(
            (16, 32, 64), (128,), (1, 2, 4, 8, 16), (4, 8), (3, 4)):
        if block_n > max(16, N * 2) or K % (block_k * split_k) or K // (block_k * split_k) < 1:
            continue
        yield block_n, block_k, split_k, warps, stages


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m', type=int, nargs='+', default=[32, 24, 8])
    p.add_argument('--copies', type=int, default=16)
    p.add_argument('--quick', action='store_true', help='check correctness and the heuristic only')
    p.add_argument('--output')
    a = p.parse_args()
    torch.manual_seed(0)
    report = {}
    for N, K in SHAPES:
        copies = [torch.randn(N, K, device='cuda', dtype=torch.bfloat16) * 0.02 for _ in range(a.copies)]
        nbytes = N * K * 2
        for M in a.m:
            x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
            ref = (x.float() @ copies[0].float().T)
            out = bf16_gemm(x, copies[0]).float()
            err = ((out - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-2, (N, K, M, err)
            cublas = graph_time(lambda w: torch.nn.functional.linear(x, w), copies)
            best = (graph_time(lambda w: bf16_gemm(x, w), copies), default_config(N, K))
            if not a.quick:
                for config in candidates(N, K):
                    try:
                        t = graph_time(lambda w, c=config: bf16_gemm(x, w, c), copies, iters=10)
                    except Exception:
                        continue
                    if t < best[0]:
                        best = (t, config)
            row = dict(cublas_us=round(cublas, 1), kernel_us=round(best[0], 1), config=best[1],
                       kernel_gbs=round(nbytes / best[0] / 1e3, 1), cublas_gbs=round(nbytes / cublas / 1e3, 1),
                       max_rel_err=err)
            report[f'{N}x{K}@M{M}'] = row
            print(f'N={N:5d} K={K:5d} M={M:2d}: cuBLAS {cublas:7.1f} us ({row["cublas_gbs"]:6.1f} GB/s)  '
                  f'kernel {best[0]:7.1f} us ({row["kernel_gbs"]:6.1f} GB/s) {best[1]}', flush=True)
        del copies
    print('DECODE_CONFIGS = {')
    for N, K in SHAPES:
        row = report.get(f'{N}x{K}@M32')
        if row and row['kernel_us'] < row['cublas_us']:
            print(f'    ({N}, {K}): {tuple(row["config"])},')
    print('}')
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
