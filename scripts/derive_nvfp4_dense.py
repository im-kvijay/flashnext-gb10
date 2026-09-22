"""Derive a checkpoint whose large dense projections are weight-only NVFP4 (W4A16_NVFP4).

The pinned NVIDIA checkpoint keeps GDN and QSA projections in BF16 (5.3 GB of the
8.6 GB read per decode step). This writes a sibling checkpoint in which the
selected projections use ModelOpt NVFP4 weights (E2M1 values, one FP8 E4M3 scale
per 16 inputs, one FP32 global scale) served by vLLM's FP4 Marlin GEMM with BF16
activations, so no activation calibration is needed. Fused runtime layers
(in_proj_qkv + in_proj_z, q/k/v) share one global scale. lm_head, embeddings,
hyperconnections, shared experts, routers, the PLE table, the MTP block and all
routed experts are copied byte for byte.

Shards are rewritten by streaming: untouched tensors are copied as raw bytes, so
memory stays bounded and written pages are dropped from the page cache. Shards
without converted tensors are symlinked. The source checkpoint is never modified.

This changes numerics: its outputs require a paired quality comparison before use.
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashnext_gb10 import nvfp4  # noqa: E402

DEFAULT_TARGETS = [
    r'model\.language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)',
    r'model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',
]
DTYPES = {torch.uint8: 'U8', torch.float8_e4m3fn: 'F8_E4M3', torch.float32: 'F32'}
CHUNK = 64 << 20


def header(path):
    with open(path, 'rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        return json.loads(stream.read(size)), 8 + size


def fused_group(prefix):
    """Runtime-fused projections must share one NVFP4 global scale."""
    return re.sub(r'\.(in_proj_qkv|in_proj_z)$', '.in_proj_qkvz',
                  re.sub(r'\.(q_proj|k_proj|v_proj)$', '.qkv_proj', prefix))


def pattern_matches(pattern, prefix):
    """True when an exclude entry (exact or glob, e.g. layers.*.linear_attn*) covers this module prefix."""
    return fnmatch.fnmatchcase(prefix, pattern)


def drop_cache(fd, start, length):
    try:
        os.posix_fadvise(fd, start, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def write_shard(source, destination, converted):
    """Stream `source` to `destination`, replacing each converted prefix's BF16 weight."""
    src_header, data_start = header(source)
    replaced = {prefix + '.weight' for prefix in converted}
    entries, order, offset = {}, [], 0
    for name, info in src_header.items():
        if name == '__metadata__' or name in replaced:
            continue
        begin, end = info['data_offsets']
        entries[name] = dict(dtype=info['dtype'], shape=info['shape'], data_offsets=[offset, offset + end - begin])
        order.append(('copy', name, begin, end))
        offset += end - begin
    for prefix, tensors in converted.items():
        for suffix, tensor in tensors.items():
            name = f'{prefix}.{suffix}'
            size = tensor.numel() * tensor.element_size()
            entries[name] = dict(dtype=DTYPES[tensor.dtype], shape=list(tensor.shape), data_offsets=[offset, offset + size])
            order.append(('new', name, tensor, None))
            offset += size
    entries['__metadata__'] = src_header.get('__metadata__', {'format': 'pt'})
    blob = json.dumps(entries, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    tmp = destination.with_suffix('.partial')
    with open(source, 'rb') as reader, open(tmp, 'wb') as writer:
        writer.write(struct.pack('<Q', len(blob)) + blob)
        written = 8 + len(blob)
        for kind, name, a, b in order:
            if kind == 'copy':
                reader.seek(data_start + a)
                remaining = b - a
                while remaining:
                    piece = reader.read(min(CHUNK, remaining))
                    writer.write(piece)
                    remaining -= len(piece)
                drop_cache(reader.fileno(), data_start + a, b - a)
            else:
                writer.write(a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            if writer.tell() - written >= 1 << 30:
                writer.flush()
                os.fdatasync(writer.fileno())
                drop_cache(writer.fileno(), 0, 0)
                written = writer.tell()
        writer.flush()
        os.fdatasync(writer.fileno())
        drop_cache(writer.fileno(), 0, 0)
    tmp.rename(destination)


def main(a):
    src, dst = Path(a.source).resolve(), Path(a.destination)
    dst.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    targets = [re.compile(t + r'$') for t in (a.target or DEFAULT_TARGETS)]
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    by_shard = defaultdict(list)
    for name, shard in index['weight_map'].items():
        if name.endswith('.weight') and any(t.match(name[:-len('.weight')]) for t in targets):
            by_shard[shard].append(name[:-len('.weight')])

    def load(prefix):
        with safe_open(src / index['weight_map'][prefix + '.weight'], framework='pt') as stream:
            tensor = stream.get_tensor(prefix + '.weight')
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f'{prefix}.weight is {tensor.dtype}, expected BF16')
        return tensor

    # Pass 1: one global amax per runtime-fused group.
    amax = {}
    for prefixes in by_shard.values():
        for prefix in prefixes:
            group = fused_group(prefix)
            value = load(prefix).to(device).float().abs().max().item()
            amax[group] = max(amax.get(group, 0.0), value)

    converted_all, stats = [], []
    for shard in sorted(set(index['weight_map'].values())):
        prefixes = by_shard.get(shard)
        if not prefixes:
            os.symlink(src / shard, dst / shard)
            continue
        converted = {}
        for prefix in sorted(prefixes):
            weight = load(prefix).to(device)
            scale_2 = nvfp4.global_scale(torch.tensor(amax[fused_group(prefix)], device=device))
            packed, scale, scale_2 = nvfp4.quantize(weight, scale_2, search=not a.no_search)
            error = (nvfp4.dequantize(packed, scale, scale_2) - weight.float()).norm() / weight.float().norm()
            stats.append((prefix, error.item()))
            converted[prefix] = {'weight': packed.cpu(), 'weight_scale': scale.cpu(),
                                 'weight_scale_2': scale_2.reshape(()).cpu()}
            for suffix in ('weight_scale', 'weight_scale_2'):
                index['weight_map'][f'{prefix}.{suffix}'] = shard
            converted_all.append(prefix)
            del weight
        write_shard(src / shard, dst / shard, converted)
        print(f'{shard}: converted {len(prefixes)} projections', flush=True)

    for item in src.iterdir():
        if item.name in {'config.json', 'hf_quant_config.json', 'model.safetensors.index.json'}:
            continue
        if not (dst / item.name).exists():
            (os.symlink if item.suffix == '.safetensors' or item.is_dir() else shutil.copy2)(item, dst / item.name)
    (dst / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2))

    entry = {'quant_algo': 'W4A16_NVFP4', 'group_size': nvfp4.BLOCK}
    converted_set = set(converted_all)
    covered = lambda pattern: any(pattern_matches(pattern, p) for p in converted_all)
    # Dense modules previously covered by a wildcard stay excluded explicitly.
    keep = sorted({name.rsplit('.', 1)[0] for name in index['weight_map']
                   if re.match(r'model\.language_model\.layers\.\d+\.(linear_attn|self_attn|mlp\.shared_expert)', name)
                   and name.endswith('.weight') and name.rsplit('.', 1)[0] not in converted_set})
    config = json.loads((src / 'config.json').read_text())
    quant = config['quantization_config']
    quant['ignore'] = [p for p in quant['ignore'] if not covered(p)] + keep
    for prefix in converted_all:
        quant['quantized_layers'][prefix] = entry
    (dst / 'config.json').write_text(json.dumps(config, indent=2))
    hf = json.loads((src / 'hf_quant_config.json').read_text())
    hq = hf['quantization']
    hq['exclude_modules'] = [p for p in hq['exclude_modules'] if not covered(p)] + keep
    for prefix in converted_all:
        hq['quantized_layers'][prefix] = entry
    (dst / 'hf_quant_config.json').write_text(json.dumps(hf, indent=2))
    errors = [e for _, e in stats]
    report = dict(source=str(src), method='W4A16_NVFP4, FP8 E4M3 per-16 scales, fused-group global scale',
                  scale_search=not a.no_search, converted=len(converted_all),
                  targets=[t.pattern for t in targets],
                  relative_error_mean=sum(errors) / max(len(errors), 1),
                  relative_error_max=max(errors, default=0.0),
                  worst=sorted(stats, key=lambda s: -s[1])[:5])
    (dst / 'derivation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('converted', 'relative_error_mean', 'relative_error_max')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('destination')
    p.add_argument('--target', action='append', help='regex for a module prefix; repeatable')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-search', action='store_true', help='plain absmax block scales')
    main(p.parse_args())
