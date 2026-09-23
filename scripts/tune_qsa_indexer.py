"""Tune the QSA indexer's decode scoring kernel launch on the served 8 x 200k shape.

Eight requests, MTP-3 (four query rows each), four BF16 index heads of 128,
compress ratio 4 (50k visible compressed rows of a 53,248-column table),
one cache per QSA layer (12) so reads come from DRAM, CUDA graph timing. The
kernel is vLLM's; only launch parameters vary. Checks that every config
selects the same blocks as vLLM's default launch.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer  # noqa: E402
from flashnext_gb10.qsa_decode_config import select_paged_decode  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', type=int, default=8)
    p.add_argument('--query-len', type=int, default=4)
    p.add_argument('--visible', type=int, default=50000, help="visible compressed rows")
    p.add_argument('--columns', type=int, default=53248)
    p.add_argument('--page', type=int, default=64)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--output')
    a = p.parse_args()
    torch.manual_seed(0)
    dev = 'cuda'
    R, Q = a.requests, a.query_len
    width = a.columns // a.page
    blocks = R * width + 1
    caches = [torch.nn.functional.normalize(torch.randn(blocks, a.page, 1, 128, device=dev), dim=-1).bfloat16()
              for _ in range(a.layers)]
    table = (torch.randperm(blocks - 1, device=dev)[:R * width] + 1).view(R, width).to(torch.int32)
    q = torch.nn.functional.normalize(torch.randn(R * Q, 4, 128, device=dev), dim=-1).bfloat16()
    visible = torch.full((R * Q,), a.visible, dtype=torch.int32, device=dev)
    visible += torch.arange(Q, device=dev, dtype=torch.int32).repeat(R)
    topk, ratio = 2048, 4

    def out():
        return torch.empty(R * Q, topk // ratio, dtype=torch.int32, device=dev)

    ref = out()
    qsa_indexer.qsa_select_paged_decode(q, caches[0], table, visible, topk, ratio, Q, ref)
    ref_sorted = ref.sort(-1).values

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / (10 * a.layers)

    outs = [out() for _ in caches]
    base = timed(lambda: [qsa_indexer.qsa_select_paged_decode(q, c, table, visible, topk, ratio, Q, o)
                          for c, o in zip(caches, outs)])
    key_bytes = R * a.visible * 128 * 2
    print(f'vLLM default: {base:.1f} us per layer (keys {key_bytes / 1e6:.1f} MB, {key_bytes / base / 1e3:.0f} GB/s)',
          flush=True)
    results = {'default_us': base}
    best = (base, None)
    for block_n, warps, stages, tiles in itertools.product((64, 128, 256), (1, 2, 4, 8), (2, 3, 4), (1, 2, 4)):
        config = (block_n, warps, stages, tiles)
        try:
            chk = out()
            select_paged_decode(qsa_indexer, config, q, caches[0], table, visible, topk, ratio, Q, chk)
            if not torch.equal(chk.sort(-1).values, ref_sorted):
                print(config, 'selection differs; skipped')
                continue
            t = timed(lambda c=config: [select_paged_decode(qsa_indexer, c, q, cc, table, visible, topk, ratio, Q, o)
                                        for cc, o in zip(caches, outs)])
        except Exception as e:
            print(config, 'failed', type(e).__name__)
            continue
        results[','.join(map(str, config))] = t
        if t < best[0]:
            best = (t, config)
            print(f'{config}: {t:.1f} us per layer', flush=True)
    print(f'best {best[1]}: {best[0]:.1f} us vs default {base:.1f} us '
          f'({(base - best[0]) * a.layers / 1000:.2f} ms per step over {a.layers} layers)')
    if best[1] is not None:
        results['best'] = ','.join(map(str, best[1]))
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
