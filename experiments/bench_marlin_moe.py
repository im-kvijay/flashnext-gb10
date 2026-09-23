"""Time vLLM's NVFP4 Marlin MoE (W4A16) at decode batch sizes against weight bytes.

Shapes follow the served checkpoint: 512 experts, hidden 2560, intermediate
640, top-10. Routing is synthetic with a controlled number of distinct experts
per call, so the effective bandwidth for a given distinct-expert count can be
compared with the profiled 1.35 ms per layer. Weight values do not matter for
timing; one expert is packed properly and its bytes are replicated.
"""
import argparse
import json

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_nvfp4_like
from vllm.scalar_type import scalar_types


def packed(e, n, k):
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) / 10
    _, q, s, g = rand_marlin_weight_nvfp4_like(w, 16)
    q = q.unsqueeze(0).repeat(e, *([1] * q.dim()))
    q.view(torch.uint8).random_(0, 256)
    return q, s.unsqueeze(0).repeat(e, *([1] * s.dim())), g.reshape(1).repeat(e)


def routing(m, e, top_k, distinct, gen):
    """m tokens, top_k distinct experts each, drawn from `distinct` experts, all of which are used."""
    pool = torch.randperm(e, generator=gen)[:distinct]
    ids = torch.empty(m, top_k, dtype=torch.int64)
    order = torch.randperm(distinct, generator=gen)
    cursor = 0
    for t in range(m):
        chosen = []
        while len(chosen) < top_k:
            if cursor < distinct:
                x = int(pool[order[cursor]]); cursor += 1
            else:
                x = int(pool[torch.randint(distinct, (1,), generator=gen)])
            if x not in chosen:
                chosen.append(x)
        ids[t] = torch.tensor(chosen)
    return ids.to(torch.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experts', type=int, default=512)
    p.add_argument('--hidden', type=int, default=2560)
    p.add_argument('--inter', type=int, default=640)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--m', type=int, nargs='+', default=[32, 24, 8])
    p.add_argument('--distinct', type=int, nargs='+', default=[40, 80, 120, 160, 200, 240])
    p.add_argument('--output')
    a = p.parse_args()
    E, H, I = a.experts, a.hidden, a.inter
    w1, s1, g1 = packed(E, 2 * I, H)
    w2, s2, g2 = packed(E, H, I)
    per_expert = sum(t[0].numel() * t[0].element_size() for t in (w1, s1, w2, s2))
    print(f'packed bytes per expert {per_expert / 1e6:.3f} MB', flush=True)
    gen = torch.Generator().manual_seed(0)
    report = []
    for m in a.m:
        x = torch.randn(m, H, device='cuda', dtype=torch.bfloat16)
        for u in a.distinct:
            if u > m * a.top_k or u < a.top_k:
                continue
            sets = [routing(m, E, a.top_k, u, gen).cuda() for _ in range(8)]
            weights = torch.softmax(torch.randn(m, a.top_k, device='cuda'), -1)

            def call(ids):
                return fused_marlin_moe(x, w1, w2, None, None, s1, s2, weights, ids,
                                        quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E,
                                        global_scale1=g1, global_scale2=g2)
            for ids in sets:
                call(ids)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for ids in sets:
                    call(ids)
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            iters = 20
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000 / (iters * len(sets))
            gbs = u * per_expert / us / 1e3
            report.append(dict(m=m, distinct=u, us=round(us, 1), gbs=round(gbs, 1)))
            print(f'M={m:3d} distinct={u:3d}: {us:8.1f} us per layer, {gbs:6.1f} GB/s', flush=True)
    if a.output:
        open(a.output, 'w').write(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
