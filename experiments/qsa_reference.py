"""FP32 transformers attention versus vLLM's QSA layer on captured inputs.

Uses flashnext_gb10/qsa_capture.py records of a first prefill chunk
(positions from 0, below the indexer budget, so the reference is dense causal
attention). Reports the relative error of vLLM's output per position bucket;
an error that grows with position points at the long-sequence attention path.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--capture', required=True, help='one qsa_capture .pt file whose positions start at 0')
    a = p.parse_args()
    record = torch.load(a.capture)
    layer = int(record['layer'].split('layers.')[1].split('.')[0])
    model = Path(a.model)
    raw = json.loads((model / 'config.json').read_text())
    config = Qwen4ExpTextConfig(**raw.get('text_config', raw))
    config._attn_implementation = 'eager'
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.language_model.layers.{layer}.self_attn.'
    attention = M.Qwen4ExpTextAttention(config, layer)
    state = {}
    for name in [k for k in index if k.startswith(prefix)]:
        with safe_open(str(model / index[name]), 'pt') as f:
            state[name[len(prefix):]] = f.get_tensor(name)
    attention.load_state_dict(state)
    attention = attention.cuda().float().eval()
    attention.indexer.forward = lambda h, pe, mask, cache: torch.ones_like(mask, dtype=torch.bool)
    positions = record['positions'].long().cuda()
    assert positions[0] == 0 and positions[-1] < config.indexer_budget
    x = record['hidden'].float().cuda()[None]
    T = x.shape[1]
    rotary = M.Qwen4ExpTextRotaryEmbedding(config, device='cuda')
    cos_sin = rotary(x, positions[None, None].expand(3, 1, -1))  # text: identical t/h/w positions
    mask = torch.zeros(1, 1, T, T, device='cuda').masked_fill(~torch.ones(T, T, dtype=torch.bool, device='cuda').tril(), float('-inf'))
    with torch.no_grad():
        ref, _ = attention(x, cos_sin, attention_mask=mask)
    ref = ref[0]
    served = record['output'].float().cuda()
    rel = (served - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
    print(f"{Path(a.capture).name}: layer {layer} T={T} served vs FP32 mean {rel.mean():.2e} median {rel.median():.2e} max {rel.max():.2e}")
    for lo in range(0, T, 256):
        print(f'   positions {lo:5d}-{min(T, lo + 256) - 1:5d}: mean {rel[lo:lo + 256].mean():.2e} max {rel[lo:lo + 256].max():.2e}')


if __name__ == '__main__':
    main()
