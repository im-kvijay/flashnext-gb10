"""Re-run upstream allocation grouping on an actual saved layout receipt."""
import argparse
from dataclasses import asdict
import json
from types import SimpleNamespace as NS

import torch
from vllm.v1.core.kv_cache_utils import _get_packed_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec, FullAttentionSpec, KVQuantMode, MLAAttentionSpec, MambaSpec,
)

from flashnext_gb10.cache_packing import normalize_single_device_specs

p=argparse.ArgumentParser()
p.add_argument('--layout',required=True)
a=p.parse_args()
receipt=json.load(open(a.layout))
specs={}
for group in receipt['layout']['kv_cache_groups']:
    for name, raw in group['kv_cache_spec']['kv_cache_specs'].items():
        raw=dict(raw)
        if 'shapes' in raw:
            cls=MambaSpec
            raw['shapes']=tuple(tuple(shape) for shape in raw['shapes'])
            raw['dtypes']=tuple(getattr(torch,dtype.split('.')[-1]) for dtype in raw['dtypes'])
        else:
            cls=(CircularBufferSpec if name.endswith('raw_key_cache') else
                 MLAAttentionSpec if name.endswith('compressed_key_cache') else FullAttentionSpec)
            raw['dtype']=getattr(torch,raw['dtype'].split('.')[-1])
            raw['kv_quant_mode']=KVQuantMode(raw['kv_quant_mode'])
        specs[name]=cls(**raw)
config=NS(parallel_config=NS(tensor_parallel_size=1,decode_context_parallel_size=1,
                            prefill_context_parallel_size=1),kv_transfer_config=None,
          model_config=NS(hf_text_config=NS(model_type='qwen4_exp_text'),max_model_len=212992),
          cache_config=NS(mamba_cache_mode='align',get_resolved_kv_cache_layout=lambda:NS(is_block_outermost=True)),
          speculative_config=NS(method='mtp',use_eagle=lambda:True,use_eagle_block_drop=lambda:True))
normalized=normalize_single_device_specs(config,specs)
for name,spec in specs.items():
    before,after=asdict(spec),asdict(normalized[name])
    changed={k for k in before if before[k]!=after[k]}
    assert changed==({'tp_replicated'} if name.endswith('.ple') else set())
old=_get_packed_kv_cache_groups(config,specs)
new=_get_packed_kv_cache_groups(config,normalized)
assert len(new)==len(old)-1
assert sorted(n for g in new for n in g.layer_names)==sorted(specs)
def blocks(groups):
    return sum((g.kv_cache_spec.max_memory_usage_bytes(config)+g.kv_cache_spec.page_size_bytes-1)//g.kv_cache_spec.page_size_bytes for g in groups)
assert blocks(new)<blocks(old)
assert max(g.kv_cache_spec.page_size_bytes for g in new)<=max(g.kv_cache_spec.page_size_bytes for g in old)
config.parallel_config.tensor_parallel_size=2
try:
    normalize_single_device_specs(config,specs)
except ValueError:
    pass
else:
    raise AssertionError('TP2 must be refused')
print(json.dumps({'status':'PASS','old_groups':len(old),'new_groups':len(new),
                  'old_blocks_per_request':blocks(old),'new_blocks_per_request':blocks(new)}))
