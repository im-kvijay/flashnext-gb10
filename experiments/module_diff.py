"""Compare module captures of two identical requests (flashnext_gb10/module_capture.py).

Warmup forwards use max-batch shapes; the first real chunk shape that occurs
at least twice for a module is paired (run 1 vs run 2 of the same request).
"""
import collections
import re
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
by_module = collections.defaultdict(list)
for path in sorted(root.glob('*.pt')):
    name, index = re.match(r'(.+)\.(\d+)\.pt$', path.name).groups()
    by_module[name].append(path)


def rel(a, b):
    a, b = a.float().flatten(1), b.float().flatten(1)
    return ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6))


def key(name):
    m = re.match(r'L(\d+)(.*)', name)
    order = {'': 9, '.ple': 1, '.linear_attn': 2, '.self_attn': 2, '.mlp.gate': 3, '.mlp': 4}
    return int(m.group(1)), order.get(m.group(2), 5)


for name in sorted(by_module, key=key):
    records = [torch.load(p) for p in by_module[name]]
    shapes = [tuple(r['output'].shape) for r in records]
    counts = collections.Counter(shapes)
    pair = next((s for s in shapes if counts[s] >= 2 and s[0] != max(x[0] for x in shapes)), None) \
        or next((s for s in shapes if counts[s] >= 2), None)
    if pair is None:
        print(f'{name}: no repeated shape {shapes}')
        continue
    a, b = [r for r, s in zip(records, shapes) if s == pair][:2]
    d = rel(a['output'], b['output'])
    line = f'{name:18s} T={pair[0]:5d} out rel mean {d.mean():.2e} max {d.max():.2e} first>1e-2 {(d > 1e-2).nonzero()[:1].flatten().tolist()}'
    if a['input'] is not None and b['input'] is not None:
        di = rel(a['input'], b['input'])
        line += f' | in rel mean {di.mean():.2e} max {di.max():.2e}'
    if name.endswith('mlp.gate'):
        ta = a['output'].float().topk(10, dim=-1).indices.sort(-1).values
        tb = b['output'].float().topk(10, dim=-1).indices.sort(-1).values
        line += f' | top10 set differs {(ta != tb).any(-1).float().mean():.3f}'
    print(line)
