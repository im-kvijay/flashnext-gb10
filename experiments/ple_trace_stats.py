"""Summarize FLASHNEXT_PLE_TRACE files: decode-step gather time, rows, reuse.

Each record: <count, heads, gather_ns, wall_ns> then count*heads int64 row IDs.
Rows are 160-byte FP8 rows of one flat table, so a row's 4 KiB page is
id * 160 // 4096. Reuse is measured against every earlier access (prefill
included), an upper bound on what the page cache can serve.
"""
import argparse
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np


def records(path):
    data = Path(path).read_bytes()
    offset = 0
    while offset + 32 <= len(data):
        count, heads, gather_ns, wall_ns = struct.unpack_from('<qqqq', data, offset)
        offset += 32
        ids = np.frombuffer(data, dtype=np.int64, count=count * heads, offset=offset)
        offset += count * heads * 8
        yield count, heads, gather_ns, wall_ns, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('traces', nargs='+')
    p.add_argument('--decode-max', type=int, default=64, help='largest token count treated as a decode step')
    p.add_argument('--row-bytes', type=int, default=160)
    p.add_argument('--direct-mapped-gib', default='2,4,8',
                   help='direct-mapped cache sizes to simulate, both as 4 KiB pages and as 160-byte rows')
    p.add_argument('--lru-gib', default='', help='comma-separated LRU sizes to simulate (slow on long prefills)')
    a = p.parse_args()
    for path in a.traces:
        steps = list(records(path))
        pages_all = [(r[4] * a.row_bytes // 4096) for r in steps]
        decode = [i for i, r in enumerate(steps) if r[0] <= a.decode_max]
        print(f'{path}: {len(steps)} records, {len(decode)} decode steps')
        if not decode:
            continue
        ms = np.array([steps[i][2] / 1e6 for i in decode])
        rows = np.array([steps[i][0] * steps[i][1] for i in decode])
        print(f'  decode gather ms: mean {ms.mean():.1f} p50 {np.median(ms):.1f} p90 {np.percentile(ms, 90):.1f} '
              f'max {ms.max():.1f}; rows/step mean {rows.mean():.0f}')
        # Late decode (last half) is the steady state once all streams decode.
        late = decode[len(decode) // 2:]
        print(f'  late-half gather ms mean {np.mean([steps[i][2] / 1e6 for i in late]):.1f}')
        seen = set()
        first_decode = decode[0]
        reuse = []
        for i, pages in enumerate(pages_all):
            unique = np.unique(pages)
            if i >= first_decode and steps[i][0] <= a.decode_max:
                reuse.append(np.mean([pg in seen for pg in unique.tolist()]))
            seen.update(unique.tolist())
        print(f'  decode pages already touched earlier (infinite cache): {np.mean(reuse):.3f}')
        decode_flags = [i >= first_decode and steps[i][0] <= a.decode_max for i in range(len(steps))]
        for gib in [float(x) for x in a.direct_mapped_gib.split(',') if x]:
            for unit, keys, entry_bytes in (('page', pages_all, 4096),
                                            ('row', [r[4] for r in steps], a.row_bytes + 8)):
                slots = int(gib * 2**30 // entry_bytes)
                tags = np.full(slots, -1, dtype=np.int64)
                hits = total = 0
                for flag, k in zip(decode_flags, keys):
                    k = np.unique(k)
                    slot = k % slots
                    hit = tags[slot] == k
                    if flag:
                        hits += int(hit.sum())
                        total += k.size
                    tags[slot] = k
                print(f'  direct-mapped {gib:g} GiB of {unit}s ({slots / 1e6:.1f}M entries): '
                      f'decode {unit} hit rate {hits / max(total, 1):.3f}')
        for gib in [float(x) for x in a.lru_gib.split(',') if x]:
            capacity = int(gib * 2**30 // 4096)
            lru = OrderedDict()
            hits = total = 0
            for i, pages in enumerate(pages_all):
                is_decode = i >= first_decode and steps[i][0] <= a.decode_max
                for pg in np.unique(pages).tolist():
                    if pg in lru:
                        lru.move_to_end(pg)
                        hits += is_decode
                    else:
                        lru[pg] = None
                        if len(lru) > capacity:
                            lru.popitem(last=False)
                    total += is_decode
            print(f'  LRU {gib:g} GiB page cache: decode hit rate {hits / max(total, 1):.3f}')


if __name__ == '__main__':
    main()
