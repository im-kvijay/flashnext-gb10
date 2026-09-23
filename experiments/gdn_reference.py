"""FP32 transformers Gated DeltaNet layer versus vLLM's captured linear_attn.

Uses flashnext_gb10/module_capture.py records (keyword inputs captured) of a
first prefill chunk; the transformers module runs from a zero state, which is
correct for a chunk starting at position 0. Reports relative error by position.
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--captures', required=True)
    p.add_argument('--layer', type=int, default=1)
    a = p.parse_args()
    model = Path(a.model)
    raw = json.loads((model / 'config.json').read_text())
    config = Qwen4ExpTextConfig(**raw.get('text_config', raw))
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.language_model.layers.{a.layer}.linear_attn.'
    gdn = M.Qwen4ExpTextGatedDeltaNet(config, a.layer)
    state = {}
    for name in [k for k in index if k.startswith(prefix)]:
        with safe_open(str(model / index[name]), 'pt') as f:
            state[name[len(prefix):]] = f.get_tensor(name)
    missing, unexpected = gdn.load_state_dict(state, strict=False)
    print('missing', missing, 'unexpected', unexpected)
    gdn = gdn.cuda().float().eval()
    name = f'L{a.layer:02d}.linear_attn'
    records = [torch.load(q) for q in sorted(Path(a.captures).glob(name + '.[0-9][0-9][0-9].pt'))]
    shapes = [r['output'].shape[0] for r in records]
    counts = collections.Counter(shapes)
    size = next(s for s in shapes if counts[s] >= 2 and s != max(shapes))
    record = next(r for r, s in zip(records, shapes) if s == size)
    if record['input'] is None:
        raise SystemExit('capture has no input; rerun with keyword-input capture')
    x = record['input'].float().cuda()[None]
    with torch.no_grad():
        ref = gdn(x)[0]
    served = record['output'].float().cuda()
    rel = (served - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
    T = rel.shape[0]
    print(f'{name}: T={T} served vs FP32 mean {rel.mean():.2e} median {rel.median():.2e} max {rel.max():.2e}')
    for lo in range(0, T, 256):
        print(f'   positions {lo:5d}-{min(T, lo + 256) - 1:5d}: mean {rel[lo:lo + 256].mean():.2e} max {rel[lo:lo + 256].max():.2e}')


if __name__ == '__main__':
    main()
