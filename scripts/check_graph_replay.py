"""Exercise the actual NVMe plugin's CPU/GPU handoff across graph replays.

A small stand-in for vLLM's distributed constructor lets the test use a known
table. Lookup, staging, prefetch, finalization and graph decorators are production
code; expected values come from direct GPU indexing of the resident source.
"""
import logging
import os
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
import tempfile
from types import SimpleNamespace

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpPLEPinnedHostEmbedding
from flashnext_gb10.vllm_nvme import make_nvme_embedding


class Constructor(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        torch.nn.Module.__init__(self)
        self.tp_size = self.etp_data_parallel_size = 1
        self.embedding_dim = embedding_dim
        self.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=num_embeddings)
        self.weight = torch.nn.Parameter(self.allocate_embedding_weight(
            num_embeddings, embedding_dim, torch.float8_e4m3fn), requires_grad=False)

    def weight_loader(self, param, value, checkpoint_start=None):
        start = checkpoint_start or 0
        param.data[start:start + value.shape[0]].copy_(value)


torch.manual_seed(7231)
with tempfile.TemporaryDirectory() as d:
    upstream = SimpleNamespace(Qwen4ExpPLEEmbedding=Constructor,
                               Qwen4ExpPLEPinnedHostEmbedding=Qwen4ExpPLEPinnedHostEmbedding,
                               logger=logging.getLogger("nvme-test"))
    cls = make_nvme_embedding(upstream, d)
    layer = cls(8192, 160, params_dtype=torch.bfloat16, padding_size=128,
                prefix="test", embedding_method=None, num_ngram_heads=16, max_total_tokens=8)
    source = torch.randn(8192, 160, device="cpu").to(torch.float8_e4m3fn)
    layer.weight_loader(layer.weight, source)
    resident = source.view(torch.uint8).to("cuda")
    ids = torch.randint(0, 8192, (8, 16), device="cuda")
    hidden = torch.zeros(8, 2560, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        layer.start_prefetch(hidden, ids)
        output = layer(hidden)
        torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with BreakableCUDAGraphCapture() as graph:
            layer.start_prefetch(hidden, ids)
            output = layer(hidden)
            observable = output.view(torch.uint8).clone()
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(32):
        ids.copy_(torch.randint(0, 8192, ids.shape, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected = resident[ids].flatten(-2)
        assert torch.equal(observable, expected), f"stale or corrupt PLE rows at graph replay {step}"
    print(f"PASS: 32 distinct CUDA graph replays, 8 agents; {graph.num_eager_breaks} eager boundaries")
