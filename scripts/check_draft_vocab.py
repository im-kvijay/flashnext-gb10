"""Check draft vocabulary mapping against full logits and changing graph inputs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from flashnext_gb10.draft_vocab import register_draft_vocab

torch.manual_seed(7291)
ids = sorted({0, 247999, 248319, *range(33, 248320, 3941)})
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'vocab.json'
    path.write_text(json.dumps({
        'model_revision': 'fc694b54fb0174e0913e6adf86691ef85a4ead47',
        'token_ids': ids,
    }))
    with patch('vllm.distributed.get_tp_group', return_value=SimpleNamespace(world_size=1)):
        register_draft_vocab(str(path))
        from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP

        draft = object.__new__(Qwen4ExpMTP)
        torch.nn.Module.__init__(draft)
        # A strided full-size head avoids allocating a second model-sized matrix.
        # Each vocabulary row is distinct; full and reduced projections use the
        # same values. Compare IDs after selecting logits in target-ID order.
        backing = torch.randn(248320 + 2560, device='cuda', dtype=torch.bfloat16)
        weight = backing.as_strided((248320, 2560), (1, 1))
        draft.lm_head = SimpleNamespace(weight=weight)
        chosen = torch.tensor(ids, device='cuda')
        hidden = torch.randn(8, 2560, device='cuda', dtype=torch.bfloat16)
        for _ in range(3):
            draft.get_top_tokens(hidden)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = draft.get_top_tokens(hidden)
        for _ in range(16):
            hidden.copy_(torch.randn_like(hidden))
            graph.replay()
            full = F.linear(hidden, weight)
            expected = chosen[full.index_select(1, chosen).argmax(-1)]
            assert torch.equal(actual, expected), (actual, expected)
        assert draft.lm_head.weight is weight
        print('PASS: 16 changing graph replays, eight rows; full-logit subset IDs match')
