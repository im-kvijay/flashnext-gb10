"""Diagnostic: distinct routed experts per MoE layer call (eager mode only).

FLASHNEXT_COUNT_EXPERTS=<path.json> hooks each target-model router and records,
per batch token count, how many distinct experts that call's top-k selects.
Routed-expert bytes read per decode step follow from these counts. The .item()
synchronization makes this unsuitable for throughput runs or CUDA graphs.
"""
from collections import defaultdict
import json
import os

import torch


def register_expert_count(path):
    from vllm.models.qwen4_exp.nvidia import model as qwen4

    counts = defaultdict(list)
    calls = [0]
    original_init = qwen4.Qwen4ExpSparseMoeBlock.__init__

    def hook(module, inputs, output):
        logits = output[0] if isinstance(output, tuple) else output
        if torch.cuda.is_current_stream_capturing():
            return
        top_k = module._flashnext_top_k
        selected = torch.topk(logits.float(), top_k, dim=-1).indices
        counts[int(logits.shape[0])].append(int(torch.unique(selected).numel()))
        calls[0] += 1
        if calls[0] % 480 == 0:
            summary = {str(t): {'calls': len(v), 'mean_distinct': sum(v) / len(v)} for t, v in sorted(counts.items())}
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'top_k': top_k, 'by_tokens': summary}, f, indent=1)
            os.replace(tmp, path)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.gate._flashnext_top_k = int(self.experts.top_k if hasattr(self.experts, 'top_k') else 10)
        self.gate.register_forward_hook(hook)

    qwen4.Qwen4ExpSparseMoeBlock.__init__ = init
