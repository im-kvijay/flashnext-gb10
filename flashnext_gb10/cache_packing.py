"""Single-device packing of PLE and GDN states with identical slot lifetimes."""
from dataclasses import replace
from functools import wraps


def normalize_single_device_specs(config, specs):
    from vllm.v1.kv_cache_interface import MambaSpec

    parallel = config.parallel_config
    if any(getattr(parallel, name, 1) != 1 for name in (
        'tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size',
        'decode_context_parallel_size', 'prefill_context_parallel_size',
    )) or config.kv_transfer_config is not None:
        raise ValueError('PLE state packing requires one device and no KV connector')
    if config.model_config.hf_text_config.model_type != 'qwen4_exp_text':
        raise ValueError('PLE state packing is scoped to Qwen4Exp')
    normalized = dict(specs)
    for name, spec in specs.items():
        if name.endswith('.ple') and isinstance(spec, MambaSpec) and spec.tp_replicated:
            # TP-replicated versus TP-sharded is equivalent at TP=1. vLLM uses
            # this flag only to segregate packing and in the NIXL connector;
            # the latter is explicitly disallowed above. Shapes, dtypes, page
            # sizes, cache lifetime, and speculative rollback remain intact.
            normalized[name] = replace(spec, tp_replicated=False)
    return normalized


def register_cache_packing():
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils.get_kv_cache_groups
    if getattr(original, '_flashnext_packing', False):
        return

    @wraps(original)
    def get_groups(config, specs):
        return original(config, normalize_single_device_specs(config, specs))

    get_groups._flashnext_packing = True
    kv_cache_utils.get_kv_cache_groups = get_groups
