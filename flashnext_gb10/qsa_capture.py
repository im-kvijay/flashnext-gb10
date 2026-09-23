"""Capture QSA attention layer inputs, outputs and sparse selections (diagnostic).

FLASHNEXT_CAPTURE_QSA_DIR=<dir> (eager mode) saves, for the first
FLASHNEXT_CAPTURE_QSA_LIMIT forwards of each QSA layer with at least 64
tokens, the positions, the layer input and output, and the packed selection
rows (token indices plus a trailing valid count). Below the indexer budget a
correct selection covers every visible token. Outputs are unchanged.
"""
import collections
import os
from pathlib import Path

import torch


def register_qsa_capture(directory):
    from vllm.models.qwen4_exp.nvidia import qsa

    Attention = qsa.Qwen4ExpQSAAttention
    if getattr(Attention, '_flashnext_qsa_capture', False):
        return
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get('FLASHNEXT_CAPTURE_QSA_LIMIT', '6'))
    counts = collections.Counter()
    original = Attention.forward

    def forward(self, positions, hidden_states):
        result = original(self, positions, hidden_states)
        count = hidden_states.shape[0]
        if (count >= 64 and counts[self.layer_name] < limit
                and not torch.cuda.is_current_stream_capturing()):
            index = counts[self.layer_name]
            counts[self.layer_name] += 1
            torch.save(dict(layer=self.layer_name, positions=positions.reshape(-1)[:count].cpu(),
                            hidden=hidden_states.cpu(), output=result.cpu(),
                            selection=self.topk_indices_buffer[:count].cpu()),
                       out / f'{self.layer_name}.{index:03d}.pt')
        return result

    Attention.forward = forward
    Attention._flashnext_qsa_capture = True
