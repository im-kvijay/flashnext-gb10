"""FP32 reference for one NVFP4 MoE layer versus vLLM's captured output.

Dequantizes the layer's NVFP4 experts (E2M1 values, E4M3 scale per 16
elements, FP32 global scale) and evaluates router + routed experts + gated
shared expert in FP32 with BF16-free activations on the exact inputs vLLM
saw (flashnext_gb10/module_capture.py). Reports the served kernel's error
against the reference and the reference's own sensitivity between two
nearly identical inputs (run 1 and run 2 of the same request).
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


GRID = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])


def quant_act_nvfp4(x, input_scale):
    """Emulate NVFP4 activation quantization: static global scale, E4M3 scale per 16, E2M1 values (RNE)."""
    shape = x.shape
    blocks = x.reshape(*shape[:-1], -1, 16)
    block_scale = (blocks.abs().amax(-1, keepdim=True) / 6 / input_scale).to(torch.float8_e4m3fn).float()
    scale = block_scale * input_scale
    scaled = (blocks / scale.clamp_min(1e-30)).clamp(-6, 6)
    grid = GRID.to(x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    mag = scaled.abs()
    idx = torch.bucketize(mag, mids)
    # ties to even: at an exact midpoint prefer the even grid index
    tie = (mag == mids[(idx - 1).clamp_min(0)]) & (idx > 0) & ((idx % 2) == 1)
    idx = torch.where(tie, idx - 1, idx)
    q = grid[idx] * scaled.sign()
    return (q * scale).reshape(shape)


def dequant_nvfp4(packed, scale, scale2):
    lo, hi = packed & 0xF, packed >> 4
    values = torch.stack([E2M1[lo.long()], E2M1[hi.long()]], -1).flatten(-2)  # low nibble first
    return values * scale.float().repeat_interleave(16, -1) * scale2.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--captures', required=True)
    p.add_argument('--layer', type=int, default=1)
    a = p.parse_args()
    model = Path(a.model)
    config = json.loads((model / 'config.json').read_text())
    config = config.get('text_config', config)
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.language_model.layers.{a.layer}.mlp.'
    handles = {}

    def get(name):
        file = index[prefix + name]
        if file not in handles:
            handles[file] = safe_open(str(model / file), 'pt')
        return handles[file].get_tensor(prefix + name)

    dev = 'cuda'
    router = get('gate.weight').float().to(dev)
    shared = {n: get(f'shared_expert.{n}.weight').float().to(dev) for n in ('gate_proj', 'up_proj', 'down_proj')}
    shared_gate = get('shared_expert_gate.weight').float().to(dev)
    cache = {}

    def expert(e):
        if e not in cache:
            cache[e] = {n: dequant_nvfp4(get(f'experts.{e}.{n}.weight'), get(f'experts.{e}.{n}.weight_scale'),
                                         get(f'experts.{e}.{n}.weight_scale_2')).to(dev)
                        for n in ('gate_proj', 'up_proj', 'down_proj')}
            cache[e].update({n + '.input_scale': get(f'experts.{e}.{n}.input_scale').float().to(dev)
                             for n in ('gate_proj', 'down_proj')})
        return cache[e]

    top_k, norm = config['num_experts_per_tok'], config.get('norm_topk_prob', True)
    silu = torch.nn.functional.silu

    def moe(x, w4a4=False):
        x = x.float().to(dev)
        probs = (x @ router.T).softmax(-1)
        weights, experts = probs.topk(top_k, -1)
        if norm:
            weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in experts.unique().tolist():
            rows, slot = (experts == e).nonzero(as_tuple=True)
            w = expert(e)
            xe = quant_act_nvfp4(x[rows], w['gate_proj.input_scale']) if w4a4 else x[rows]
            h = silu(xe @ w['gate_proj'].T) * (xe @ w['up_proj'].T)
            if w4a4:
                h = quant_act_nvfp4(h, w['down_proj.input_scale'])
            out.index_add_(0, rows, (h @ w['down_proj'].T) * weights[rows, slot, None])
        s = silu(x @ shared['gate_proj'].T) * (x @ shared['up_proj'].T) @ shared['down_proj'].T
        return out + torch.sigmoid(x @ shared_gate.T) * s

    name = f'L{a.layer:02d}.mlp'
    records = [torch.load(q) for q in sorted(Path(a.captures).glob(name + '.[0-9][0-9][0-9].pt'))]
    shapes = [r['output'].shape[0] for r in records]
    counts = collections.Counter(shapes)
    size = next(s for s in shapes if counts[s] >= 2 and s != max(shapes))
    r1, r2 = [r for r, s in zip(records, shapes) if s == size][:2]

    def rel(x, y):
        x, y = x.float().to(dev), y.float().to(dev)
        d = (x - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-6)
        return f'mean {d.mean():.2e} p50 {d.median():.2e} max {d.max():.2e}'

    with torch.no_grad():
        ref1, ref2 = moe(r1['input']), moe(r2['input'])
        emu1, emu2 = moe(r1['input'], True), moe(r2['input'], True)
    print(f'layer {a.layer} tokens {size} top_k {top_k} norm_topk {norm}')
    print('input run1 vs run2          ', rel(r1['input'], r2['input']))
    print('served vs FP32 reference    ', rel(r1['output'], ref1), '|', rel(r2['output'], ref2))
    print('reference run1 vs run2      ', rel(ref1, ref2))
    print('served run1 vs run2         ', rel(r1['output'], r2['output']))
    print('served vs W4A4 emulation    ', rel(r1['output'], emu1), '|', rel(r2['output'], emu2))
    print('W4A4 emulation vs reference ', rel(emu1, ref1))
    print('W4A4 emulation run1 vs run2 ', rel(emu1, emu2))
    print('output norm / input norm    ', (ref1.norm(dim=-1) / r1['input'].float().to(dev).norm(dim=-1)).median().item())


if __name__ == '__main__':
    main()
