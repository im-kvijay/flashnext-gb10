"""GPU regression for anchor predictions, rejected KV rows and request isolation.

Run only after stopping the inference service. No model weights are needed.
"""
import json
from types import SimpleNamespace
import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def check(batch, k):
    maximum = 12
    capacity = maximum * (k + 1) + 64
    device = "cuda"
    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)
    def full(n, dtype=torch.int64):
        return torch.full((n,), -9, dtype=dtype, device=device)
    inputs = SimpleNamespace(input_ids=full(capacity, torch.int32), positions=full(capacity),
        query_start_loc=full(maximum + 1, torch.int32), seq_lens=full(maximum, torch.int32))
    mapping = [9, 1, 11, 3, 8, 7, 0, 6][:batch]
    positions = [200000 + i * 31 + j for i in range(batch) for j in range(k + 1)]
    target = SimpleNamespace(num_reqs=batch,
        num_scheduled_tokens=np.array([k + 1] * batch, dtype=np.int32),
        positions=tensor(positions), query_start_loc=tensor([i * (k + 1) for i in range(batch + 1)], torch.int32),
        idx_mapping=tensor(mapping, torch.int32))
    qslots, cpos, cslots = full(capacity), full(capacity), full(capacity)
    sind, spos, smap = full(maximum * k), full(maximum * k), full(maximum * k, torch.int32)
    temp, seeds = full(maximum, torch.float32), full(maximum)
    sampled = tensor([i % 2 for i in range(batch)], torch.int32)
    rejected = tensor([0] * batch, torch.int32)
    last = tensor([1000 + i for i in range(maximum)])
    prefill = tensor([2000 + i for i in range(maximum)])
    it = tensor([i / 10 for i in range(maximum)], torch.float32)
    iseeds = tensor([123 + i for i in range(maximum)])
    blocks = torch.arange(1, batch * 1024 + 1, device=device, dtype=torch.int32).view(batch, 1024)
    def run():
        prepare_dflash_inputs(inputs, qslots, cpos, cslots, sind, spos, smap,
            temp, seeds, target, sampled, rejected, last, prefill, it, iseeds,
            blocks, 256, 0, 1, 1, 248077, k, k, maximum, capacity, 262144,
            sample_from_anchor=True)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for replay in range(8):
        rejects = [(i + replay) % (k + 1) for i in range(batch)]
        rejected.copy_(tensor(rejects, torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        for i, state in enumerate(mapping):
            valid = k + 1 - rejects[i]
            anchor = positions[i * (k + 1)] + valid
            query = list(range(anchor, anchor + k))
            section = slice(i * k, (i + 1) * k)
            assert inputs.positions[section].tolist() == query
            assert inputs.input_ids[section].tolist() == [
                (1000 if i % 2 else 2000) + state, *([248077] * (k - 1))]
            assert sind[section].tolist() == list(range(i * k, (i + 1) * k))
            assert spos[section].tolist() == [v + 1 for v in query]
            assert smap[section].tolist() == [state] * k
            expected_slots = [(i * 1024 + p // 256 + 1) * 256 + p % 256 for p in query]
            assert qslots[section].tolist() == expected_slots
            context = slice(i * (k + 1), (i + 1) * (k + 1))
            expected_pos = positions[i * (k + 1):i * (k + 1) + valid] + [0] * rejects[i]
            assert cpos[context].tolist() == expected_pos
            expected_ctx = [(i * 1024 + p // 256 + 1) * 256 + p % 256
                            for p in expected_pos[:valid]] + [PAD_SLOT_ID] * rejects[i]
            assert cslots[context].tolist() == expected_ctx
            assert seeds[state].item() == 123 + state
        assert smap[batch * k:].tolist() == [-1] * ((maximum - batch) * k)
        assert qslots[batch * k:].tolist() == [PAD_SLOT_ID] * (capacity - batch * k)
    return {"batch": batch, "K": k, "changing_graph_replays": 8, "positions_over_200k": True}


if __name__ == "__main__":
    with torch.inference_mode():
        rows = [check(batch, k) for batch in (1, 8) for k in (4, 5, 7)]
    print(json.dumps({"passed": True, "checks": rows, "scope": "input preparation only"}, indent=2))
