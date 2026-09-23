"""Replay recorded decode-step PLE lookups against the real checkpoint shards.

Times each decode step's gather through CheckpointShards with the current
settings (FLASHNEXT_PLE_IO, FLASHNEXT_PLE_IO_THREADS, FLASHNEXT_PLE_ROW_CACHE_GIB).
The file's page cache is dropped first, so absolute times are for a cold
cache; compare modes against each other, not against the served run.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flashnext_gb10.checkpoint_shards import CheckpointShards  # noqa: E402
from ple_trace_stats import records  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--trace', required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--decode-max', type=int, default=64)
    p.add_argument('--build-dir', default='/tmp/flashnext-replay')
    a = p.parse_args()
    prefix = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_'
    path = Path(a.model) / 'model-fp8-mtp-ple.safetensors'
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    reader = safe_open(str(path), framework='pt', device='cpu')
    names = sorted((k for k in reader.keys() if k.startswith(prefix)), key=lambda k: int(k[len(prefix):].split('.')[0]))
    Path(a.build_dir).mkdir(parents=True, exist_ok=True)
    table = CheckpointShards(160, torch.float8_e4m3fn, a.build_dir)
    start = 0
    for name in names:
        tensor = reader.get_tensor(name)
        table.add(start, tensor)
        start += tensor.shape[0]
    steps = [r for r in records(a.trace) if r[0] <= a.decode_max][:a.steps]
    times = []
    for count, heads, _, _, ids in steps:
        ids = torch.from_numpy(ids.copy()).reshape(count, heads)
        out = torch.empty(count * heads * 160, dtype=torch.uint8)
        t = time.perf_counter()
        table.gather_into(ids, out, start)
        times.append((time.perf_counter() - t) * 1e3)
    times = np.array(times)
    print(f"io={os.environ.get('FLASHNEXT_PLE_IO', 'buffered')} threads={os.environ.get('FLASHNEXT_PLE_IO_THREADS', '32')} "
          f"row_cache={os.environ.get('FLASHNEXT_PLE_ROW_CACHE_GIB', '0')}: {len(times)} steps, mean {times.mean():.1f} ms, "
          f"p50 {np.median(times):.1f}, p90 {np.percentile(times, 90):.1f}, second half {times[len(times)//2:].mean():.1f}")
    if table._row_cache:
        hits, misses, unique, _ = table.row_cache_stats()
        print(f'  row cache hits {hits}, misses {misses}, unique reads {unique}')


if __name__ == '__main__':
    main()
