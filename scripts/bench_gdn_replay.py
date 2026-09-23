"""Time the GDN replay decode against vLLM's fused MTP op at the served decode shape.

Eight requests, four tokens each (MTP-3), 48 value heads, BF16 state, one
state tensor per GDN layer (36) so the working set exceeds L2, all inside a
CUDA graph as in serving. Steady state for the replay kernel: every request has
a live record and replays its accepted tokens.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashnext_gb10.gdn_replay import replay_decode  # noqa: E402

H, HV, K, V = 16, 48, 128, 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', type=int, default=8)
    p.add_argument('--width', type=int, default=4)
    p.add_argument('--layers', type=int, default=36)
    p.add_argument('--accepted', type=int, default=3)
    a_ = p.parse_args()
    dev = 'cuda'
    B, W, L = a_.requests, a_.width, a_.layers
    slots = 1 + B * W
    rows = (torch.arange(B * W, dtype=torch.int32, device=dev) + 1).view(B, W)
    cu = torch.arange(0, B * W + 1, W, dtype=torch.int32, device=dev)
    T = B * W
    qkv = torch.randn(T, 2 * H * K + HV * V, device=dev).bfloat16()
    a = torch.randn(T, HV, device=dev).bfloat16()
    b = torch.randn(T, HV, device=dev).bfloat16()
    gate = torch.randn(T, HV, V, device=dev).bfloat16()
    A_log = torch.rand(HV, device=dev) - 2
    dt_bias = torch.randn(HV, device=dev) * 0.5
    norm_w = torch.ones(V, device=dev, dtype=torch.bfloat16)
    states = [(torch.randn(slots, HV, V, K, device=dev) * 0.05).bfloat16() for _ in range(L)]
    out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
    ones = torch.ones(B, dtype=torch.int32, device=dev)
    acc = torch.full((B,), a_.accepted, dtype=torch.int32, device=dev)
    scale = K ** -0.5

    def ours():
        for s in states:
            replay_decode(qkv, a, b, A_log, dt_bias, rows, cu, acc, s, gate, norm_w, out, H, scale, 1e-6, False)

    def theirs():
        from vllm import _custom_ops as ops
        for s in states:
            ops.fused_gdn_decode_post_conv_mtp(
                mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state_indices=rows, cu_seqlens=cu,
                num_accepted_tokens=acc, state=s, output_gate=gate, norm_weight=norm_w, out=out,
                scale=scale, norm_eps=1e-6, output_gate_activation='silu')

    # Records exist after one call with accepted=1 (first slot is the current state).
    for s in states:
        replay_decode(qkv, a, b, A_log, dt_bias, rows, cu, ones, s, gate, norm_w, out, H, scale, 1e-6, False)
    per_state = B * HV * V * K * 2
    for name, fn in (('replay', ours), ('vllm', theirs)):
        try:
            fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000 / (10 * L)
            print(f'{name}: {us:.1f} us per layer, {us * L / 1000:.2f} ms per step '
                  f'({per_state / 1e6:.1f} MB state per layer per pass)', flush=True)
        except Exception as e:  # vLLM's op is unavailable off the served build
            print(f'{name}: unavailable ({type(e).__name__}: {e})'[:200])


if __name__ == '__main__':
    main()
