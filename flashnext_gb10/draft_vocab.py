"""Optional reduced MTP projection. Target logits and vocabulary are untouched."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def register_draft_vocab(path):
    from vllm.distributed import get_tp_group
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP

    payload = json.loads(Path(path).read_text())
    if payload['model_revision'] != 'fc694b54fb0174e0913e6adf86691ef85a4ead47':
        raise ValueError('draft vocabulary belongs to another checkpoint')
    ids = payload['token_ids']
    if not ids or ids != sorted(set(ids)) or min(ids) < 0 or max(ids) >= 248320:
        raise ValueError('draft vocabulary must contain unique sorted target IDs')
    if getattr(Qwen4ExpMTP, '_flashnext_draft_vocab', None) == path:
        return

    def get_top_tokens(self, hidden_states):
        if not hasattr(self, '_flashnext_draft_weight'):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('reduced draft head must be initialized by eager warmup')
            if get_tp_group().world_size != 1:
                raise ValueError('reduced draft head currently requires TP=1')
            weight = self.lm_head.weight
            if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError('reduced draft head requires an unquantized lm_head')
            if tuple(weight.shape) != (248320, 2560):
                raise ValueError(f'unexpected draft head shape {tuple(weight.shape)}')
            self._flashnext_draft_ids = torch.tensor(ids, device=weight.device, dtype=torch.long)
            self._flashnext_draft_weight = weight.index_select(0, self._flashnext_draft_ids).contiguous()
        logits = F.linear(hidden_states, self._flashnext_draft_weight)
        return self._flashnext_draft_ids[logits.argmax(dim=-1)]

    Qwen4ExpMTP.get_top_tokens = get_top_tokens
    Qwen4ExpMTP._flashnext_draft_vocab = path
