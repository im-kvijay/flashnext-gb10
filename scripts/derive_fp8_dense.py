"""Derive a checkpoint whose large dense projections are 128x128 block-scaled FP8.

The pinned NVIDIA checkpoint keeps every non-expert projection in BF16, which is
8.6 GB read per decode step. This writes a sibling checkpoint in which selected
dense linears use ModelOpt `FP8_PB_WO` (E4M3 weights, one FP32 scale per
128x128 block, dynamic activation quantization in vLLM). Routed experts, the PLE
table, embeddings, hyperconnections, norms, routers, GDN gates, the indexer,
the MTP block and lm_head are copied unchanged. Shards without converted
tensors are symlinked. The source checkpoint is never modified.

This changes numerics: its outputs require a paired quality comparison before use.
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK = 128
DEFAULT_TARGETS = [
    r'model\.language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)',
    r'model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',
    r'model\.language_model\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)',
]


def quantize_block(weight):
    out, inp = weight.shape
    if inp % BLOCK:
        raise ValueError(f'input width {inp} is not a multiple of {BLOCK}')
    pad = (-out) % BLOCK
    w = weight.float()
    if pad:
        w = torch.cat([w, w.new_zeros(pad, inp)])
    blocks = w.view(-1, BLOCK, inp // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(-1, inp)[:out].contiguous()
    return q, scale.view(-1, 1, inp // BLOCK, 1).contiguous()


def header(path):
    with open(path, 'rb') as stream:
        return json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))


def pattern_matches(pattern, prefix):
    """True when an exclude entry (exact or glob, e.g. layers.*.linear_attn*) covers this module prefix."""
    return fnmatch.fnmatchcase(prefix, pattern)


def main(a):
    src, dst = Path(a.source).resolve(), Path(a.destination)
    dst.mkdir(parents=True, exist_ok=False)
    targets = [re.compile(t + r'$') for t in (a.target or DEFAULT_TARGETS)]
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    converted = []
    for shard in sorted(set(index['weight_map'].values())):
        names = [k for k in header(src / shard) if k != '__metadata__']
        wanted = {k[:-len('.weight')] for k in names
                  if k.endswith('.weight') and any(t.match(k[:-len('.weight')]) for t in targets)}
        if not wanted:
            os.symlink(src / shard, dst / shard)
            continue
        tensors = {}
        with safe_open(src / shard, framework='pt') as stream:
            for name in names:
                tensor = stream.get_tensor(name)
                prefix = name[:-len('.weight')] if name.endswith('.weight') else None
                if prefix in wanted:
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f'{name} is {tensor.dtype}, expected BF16')
                    q, scale = quantize_block(tensor)
                    tensors[name] = q
                    tensors[prefix + '.weight_scale'] = scale
                    index['weight_map'][prefix + '.weight_scale'] = shard
                    converted.append(prefix)
                else:
                    tensors[name] = tensor
        save_file(tensors, dst / shard, metadata={'format': 'pt'})
        print(f'{shard}: converted {len(wanted)} projections', flush=True)
    for item in src.iterdir():
        if item.name in {'config.json', 'hf_quant_config.json', 'model.safetensors.index.json'}:
            continue
        if not (dst / item.name).exists():
            (os.symlink if item.suffix == '.safetensors' else shutil.copy2)(item, dst / item.name)
    (dst / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2))

    entry = {'quant_algo': 'FP8_PB_WO', 'group_size': BLOCK}
    config = json.loads((src / 'config.json').read_text())
    quant = config['quantization_config']
    covered = lambda pattern: any(pattern_matches(pattern, p) for p in converted)
    quant['ignore'] = [p for p in quant['ignore'] if not covered(p)]
    for prefix in converted:
        quant['quantized_layers'][prefix] = entry
    # Dense modules previously covered by a wildcard stay excluded explicitly.
    keep = sorted({name.rsplit('.', 1)[0] for name in index['weight_map']
                   if re.match(r'model\.language_model\.layers\.\d+\.(linear_attn|self_attn|mlp\.shared_expert)', name)
                   and name.endswith('.weight') and name.rsplit('.', 1)[0] not in set(converted)})
    quant['ignore'] += keep
    (dst / 'config.json').write_text(json.dumps(config, indent=2))
    hf = json.loads((src / 'hf_quant_config.json').read_text())
    hq = hf['quantization']
    hq['exclude_modules'] = [p for p in hq['exclude_modules'] if not covered(p)] + keep
    for prefix in converted:
        hq['quantized_layers'][prefix] = entry
    (dst / 'hf_quant_config.json').write_text(json.dumps(hf, indent=2))
    (dst / 'derivation.json').write_text(json.dumps(dict(
        source=str(src), method='FP8_PB_WO 128x128 absmax', converted=len(converted),
        targets=[t.pattern for t in targets]), indent=2))
    print(json.dumps({'converted_projections': len(converted)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('destination')
    p.add_argument('--target', action='append', help='regex for a module prefix; repeatable')
    main(p.parse_args())
