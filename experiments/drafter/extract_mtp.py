"""Copy the drafter's training inputs out of the checkpoint into a small directory.

train_mtp.py and check_equivalence.py read `--weights DIR`: the checkpoint's
mtp.* tensors plus embed_tokens and lm_head (mtp.safetensors, about 5 GB),
the target's final hyper-connection mixer (target_mixer.safetensors), and the
config/tokenizer files.

Usage: extract_mtp.py <checkpoint dir> <output dir>
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def copy_tensors(src, index, keys, path):
    by_file = {}
    for k in keys:
        by_file.setdefault(index[k], []).append(k)
    out = {}
    for f, ks in by_file.items():
        with safe_open(str(src / f), 'pt') as h:
            for k in ks:
                out[k] = h.get_tensor(k)
    save_file(out, str(path))
    return sum(v.numel() * v.element_size() for v in out.values())


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    keys = [k for k in index if k.startswith('mtp.') or k in ('lm_head.weight', 'model.language_model.embed_tokens.weight')]
    size = copy_tensors(src, index, keys, dst / 'mtp.safetensors')
    (dst / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'mtp.safetensors' for k in keys}}))
    mixer = [k for k in index if k.startswith('model.language_model.hyper_connection_mixer')]
    size += copy_tensors(src, index, mixer, dst / 'target_mixer.safetensors')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
        shutil.copy(src / name, dst / name)
    print(f'{len(keys)} drafter tensors, {len(mixer)} mixer tensors, {size / 1e9:.2f} GB -> {dst}')


if __name__ == '__main__':
    main()
