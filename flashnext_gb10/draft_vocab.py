"""Optional reduced MTP projection. Target logits and vocabulary are untouched.

FLASHNEXT_DRAFT_HEAD_W8=1 also stores the reduced projection as FP8 blocks
(one scale per 128x128) and streams it with the W8A16 decode GEMM: half the
bytes of the BF16 copy, read near bandwidth instead of through cuBLAS's GEMV
path. Only the drafter's argmax sees it; verification uses the BF16 lm_head.
"""
import json
import os
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
    w8 = os.environ.get('FLASHNEXT_DRAFT_HEAD_W8') == '1'

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
            if w8:
                from .dense_w8a16 import quantize_block_fp8
                q, scale = quantize_block_fp8(self._flashnext_draft_weight)
                self._flashnext_draft_weight = (q, scale)
        if w8:
            from .dense_w8a16 import w8a16_gemm
            x = hidden_states.reshape(-1, hidden_states.shape[-1])
            if x.dtype != torch.bfloat16 or x.stride(-1) != 1:
                x = x.to(torch.bfloat16).contiguous()
            logits = w8a16_gemm(x, *self._flashnext_draft_weight).view(*hidden_states.shape[:-1], -1)
        else:
            logits = F.linear(hidden_states, self._flashnext_draft_weight)
        return self._flashnext_draft_ids[logits.argmax(dim=-1)]

    Qwen4ExpMTP.get_top_tokens = get_top_tokens
    Qwen4ExpMTP._flashnext_draft_vocab = path
