"""Capture decoder sub-module outputs to localize nondeterminism (diagnostic).

FLASHNEXT_CAPTURE_MODULES_DIR=<dir> (eager mode) registers forward hooks on
Qwen4Exp decoder layers below FLASHNEXT_CAPTURE_MODULES_LAYERS (default 4)
and their linear_attn / self_attn / mlp / mlp.gate / ple children, saving the
first tensor output of the first FLASHNEXT_CAPTURE_MODULES_LIMIT calls with at
least 64 tokens. Comparing two identical requests shows where outputs first
differ. Outputs are unchanged.
"""
import collections
import os
from pathlib import Path

import torch


def register_module_capture(directory):
    from vllm.models.qwen4_exp.nvidia import model as nvidia_model

    Layer = nvidia_model.Qwen4ExpDecoderLayer
    if getattr(Layer, '_flashnext_module_capture', False):
        return
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    layers = int(os.environ.get('FLASHNEXT_CAPTURE_MODULES_LAYERS', '4'))
    limit = int(os.environ.get('FLASHNEXT_CAPTURE_MODULES_LIMIT', '8'))
    counts = collections.Counter()

    def hook_for(name):
        def hook(module, args, kwargs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor) or tensor.dim() == 0 or tensor.shape[0] < 64:
                return
            if torch.cuda.is_current_stream_capturing() or counts[name] >= limit:
                return
            index = counts[name]
            counts[name] += 1
            # Decoder sub-modules are called with keywords (hidden_states=...).
            first = kwargs.get('hidden_states')
            if first is None and args:
                tensors = [x for x in args if isinstance(x, torch.Tensor) and x.dim() == 2]
                first = tensors[-1] if tensors else None
            torch.save(dict(output=tensor.detach().cpu(),
                            input=first.detach().cpu() if first is not None else None),
                       out / f'{name}.{index:03d}.pt')
        return hook

    original = Layer.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.layer_idx >= layers or getattr(self, '_is_mtp', False):
            return
        base = f'L{self.layer_idx:02d}'
        self.register_forward_hook(hook_for(base), with_kwargs=True)
        for child in ('linear_attn', 'self_attn', 'mlp', 'ple'):
            module = getattr(self, child, None)
            if module is not None:
                module.register_forward_hook(hook_for(f'{base}.{child}'), with_kwargs=True)
        gate = getattr(getattr(self, 'mlp', None), 'gate', None)
        if gate is not None:
            gate.register_forward_hook(hook_for(f'{base}.mlp.gate'), with_kwargs=True)

    Layer.__init__ = __init__
    Layer._flashnext_module_capture = True
