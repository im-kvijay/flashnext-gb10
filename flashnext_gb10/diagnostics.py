"""Opt-in cache layout receipts for capacity investigations."""
from dataclasses import asdict
from functools import wraps
import json
import os
from pathlib import Path


def register_diagnostics(directory):
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.initialize_kv_cache
    if getattr(original, '_flashnext_receipt', False):
        return

    @wraps(original)
    def initialize(self, kv_cache_config, *args, **kwargs):
        result = original(self, kv_cache_config, *args, **kwargs)
        receipt = {
            'pid': os.getpid(), 'profiling': kwargs.get('is_profiling', False),
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'layout': asdict(kv_cache_config),
            'groups': [{'type': type(g.kv_cache_spec).__name__,
                        'page_bytes': g.kv_cache_spec.page_size_bytes,
                        'max_request_bytes': g.kv_cache_spec.max_memory_usage_bytes(self.vllm_config),
                        'layers': g.layer_names}
                       for g in kv_cache_config.kv_cache_groups],
        }
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / f'kv-layout-{os.getpid()}.json').write_text(json.dumps(receipt, default=str, indent=2))
        return result

    initialize._flashnext_receipt = True
    GPUModelRunner.initialize_kv_cache = initialize
