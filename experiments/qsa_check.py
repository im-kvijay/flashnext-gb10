"""Check captured QSA selections (flashnext_gb10/qsa_capture.py).

For a sequence below the indexer budget every visible token must be selected:
row p (position p) should list p + 1 distinct tokens. Prints per capture the
fraction of rows whose valid count and index set are complete.
"""
import sys
from pathlib import Path

import torch

budget = 2048
for path in sorted(Path(sys.argv[1]).glob('*.pt')):
    r = torch.load(path)
    positions, sel = r['positions'].long(), r['selection']
    counts = sel[:, -1].long()
    body = sel[:, :-1].long()
    complete = bad_count = 0
    examples = []
    for row in range(sel.shape[0]):
        p = positions[row].item()
        visible = p + 1
        n = counts[row].item()
        idx = body[row, :max(n, 0)]
        ok_count = n == min(visible, budget + 3)
        ok_set = ok_count and visible <= budget and torch.equal(idx.sort().values, torch.arange(visible))
        complete += ok_set
        if not ok_count:
            bad_count += 1
            if len(examples) < 3:
                examples.append((p, n, idx[:6].tolist(), idx[-3:].tolist() if n > 0 else []))
    print(f'{path.name}: rows {sel.shape[0]} pos {positions[0].item()}..{positions[-1].item()} '
          f'complete {complete / sel.shape[0]:.3f} wrong-count {bad_count} out-norm {r["output"].float().norm(dim=-1).mean():.2f} '
          f'examples {examples}')
