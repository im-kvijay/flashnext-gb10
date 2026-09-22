"""Check derive_nvfp4_dense.py on a small synthetic checkpoint (CPU is enough)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from flashnext_gb10 import nvfp4  # noqa: E402

torch.manual_seed(0)
L = 'model.language_model.layers'
tensors = [
    {f'{L}.0.linear_attn.in_proj_qkv.weight': torch.randn(96, 64).bfloat16() * 0.02,
     f'{L}.0.linear_attn.in_proj_a.weight': torch.randn(8, 64).bfloat16(),
     f'{L}.0.mlp.experts.0.gate_proj.weight': torch.randint(0, 255, (16, 32), dtype=torch.uint8),
     'lm_head.weight': torch.randn(50, 64).bfloat16()},
    {f'{L}.0.linear_attn.in_proj_z.weight': torch.randn(32, 64).bfloat16() * 0.5,
     f'{L}.0.linear_attn.out_proj.weight': torch.randn(64, 32).bfloat16() * 0.02,
     f'{L}.1.self_attn.q_proj.weight': torch.randn(64, 64).bfloat16() * 0.03,
     f'{L}.1.self_attn.k_proj.weight': torch.randn(16, 64).bfloat16() * 0.3,
     f'{L}.1.self_attn.v_proj.weight': torch.randn(16, 64).bfloat16() * 0.03,
     f'{L}.1.self_attn.o_proj.weight': torch.randn(64, 64).bfloat16() * 0.03,
     f'{L}.1.self_attn.indexer.index_qk_proj.weight': torch.randn(16, 64).bfloat16()},
]
with tempfile.TemporaryDirectory() as tmp:
    src, dst = Path(tmp) / 'src', Path(tmp) / 'dst'
    src.mkdir()
    weight_map = {}
    for i, group in enumerate(tensors):
        shard = f'model-0000{i + 1}-of-00002.safetensors'
        save_file(group, src / shard, metadata={'format': 'pt'})
        weight_map.update({k: shard for k in group})
    (src / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}))
    quant = {'quant_algo': 'MIXED_PRECISION', 'ignore': ['lm_head', f'{L}.*.linear_attn*', f'{L}.1.self_attn*'],
             'quantized_layers': {f'{L}.0.mlp.experts': {'quant_algo': 'NVFP4', 'group_size': 16}}}
    (src / 'config.json').write_text(json.dumps({'quantization_config': quant}))
    (src / 'hf_quant_config.json').write_text(json.dumps({'quantization': {
        'exclude_modules': list(quant['ignore']), 'quantized_layers': dict(quant['quantized_layers'])}}))
    (src / 'tokenizer.json').write_text('{}')
    subprocess.run([sys.executable, str(root / 'scripts/derive_nvfp4_dense.py'), str(src), str(dst), '--device', 'cpu'],
                   check=True)

    index = json.loads((dst / 'model.safetensors.index.json').read_text())['weight_map']
    loaded = {}
    for shard in sorted(set(index.values())):
        with safe_open(dst / shard, framework='pt') as stream:
            for name in stream.keys():
                loaded[name] = stream.get_tensor(name)
    assert set(loaded) == set(index), 'index and shard contents differ'
    converted = [k[:-len('.weight')] for group in tensors for k in group
                 if k.split('.')[-2] in ('in_proj_qkv', 'in_proj_z', 'out_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj')]
    for group in tensors:
        for name, original in group.items():
            prefix = name[:-len('.weight')]
            if prefix in converted:
                assert loaded[name].dtype == torch.uint8 and loaded[prefix + '.weight_scale'].dtype == torch.float8_e4m3fn
                decoded = nvfp4.dequantize(loaded[name], loaded[prefix + '.weight_scale'], loaded[prefix + '.weight_scale_2'])
                error = ((decoded - original.float()).norm() / original.float().norm()).item()
                assert error < 0.2, f'{prefix}: relative error {error}'
            else:
                assert torch.equal(loaded[name].view(torch.uint8), original.view(torch.uint8)), f'{name} changed'
    scale = lambda p: loaded[f'{L}.{p}.weight_scale_2'].item()
    assert scale('0.linear_attn.in_proj_qkv') == scale('0.linear_attn.in_proj_z'), 'qkv/z global scales differ'
    assert scale('1.self_attn.q_proj') == scale('1.self_attn.k_proj') == scale('1.self_attn.v_proj'), 'q/k/v global scales differ'
    assert scale('0.linear_attn.out_proj') != scale('0.linear_attn.in_proj_z')
    config = json.loads((dst / 'config.json').read_text())['quantization_config']
    for prefix in converted:
        assert config['quantized_layers'][prefix] == {'quant_algo': 'W4A16_NVFP4', 'group_size': 16}
    assert f'{L}.*.linear_attn*' not in config['ignore'] and f'{L}.0.linear_attn.in_proj_a' in config['ignore']
    assert f'{L}.1.self_attn.indexer.index_qk_proj' in config['ignore'] and 'lm_head' in config['ignore']
    assert (dst / 'tokenizer.json').exists()
    report = json.loads((dst / 'derivation.json').read_text())
    print('PASS', {k: report[k] for k in ('converted', 'relative_error_mean', 'relative_error_max')})
