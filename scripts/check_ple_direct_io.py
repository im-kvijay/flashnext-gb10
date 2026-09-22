"""O_DIRECT PLE reads must equal mapped reads byte for byte on the real checkpoint."""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from flashnext_gb10.checkpoint_shards import CheckpointShards

p = argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('--steps', type=int, default=64)
a = p.parse_args()
files = sorted(Path(a.checkpoint).glob('*.safetensors'))
readers, tensors = [], []
for path in files:
    reader = safe_open(path, framework='pt', device='cpu')
    names = sorted(k for k in reader.keys() if 'ngram' in k)
    if names:
        readers.append(reader)
        tensors += [reader.get_tensor(k) for k in names]
tensors = [t for t in tensors if t.ndim == 2 and t.dtype == torch.float8_e4m3fn]
width = tensors[0].shape[1]
directory = Path(a.checkpoint).parent / 'lookup-check'
tables = {}
for mode in ('mapped', 'direct', 'buffered'):
    table = CheckpointShards(width, torch.float8_e4m3fn, directory, io_mode=mode)
    start = 0
    for t in tensors:
        table.add(start, t)
        start += t.shape[0]
    table.seal(start)
    tables[mode] = table
rows = start
assert tables['direct'].io_mode == 'direct', 'O_DIRECT unavailable on this filesystem'
g = torch.Generator().manual_seed(4099)
boundaries = torch.tensor([p[0] for p in tables['direct'].parts][1:], dtype=torch.int64)
timing = {m: [] for m in tables}
for step in range(a.steps):
    ids = torch.randint(-3, rows + 3, (24, 16), generator=g, dtype=torch.int64)
    ids.view(-1)[:min(boundaries.numel(), 64)] = boundaries[:64] - (step % 2)
    outputs = {}
    for mode in tables:
        out = torch.empty(24 * 16 * width, dtype=torch.uint8)
        s = time.perf_counter()
        tables[mode].gather_into(ids, out, rows)
        timing[mode].append(time.perf_counter() - s)
        outputs[mode] = out
    assert torch.equal(outputs['direct'], outputs['mapped']), f'direct mismatch at step {step}'
    assert torch.equal(outputs['buffered'], outputs['mapped']), f'buffered mismatch at step {step}'
med = lambda xs: sorted(xs)[len(xs) // 2] * 1e3
print(json.dumps({'passed': True, 'steps': a.steps, 'rows_per_step': 384, 'table_rows': rows,
                  'parts': len(tensors), **{f'{m}_median_ms': med(t) for m, t in timing.items()},
                  'timing_note': 'order-dependent cache state; not a cold benchmark'}))
