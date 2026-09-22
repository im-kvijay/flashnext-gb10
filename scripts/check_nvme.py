"""Test byte-exact FP8 row gathering and stale-state handling on the GPU."""
import json
import tempfile

import torch

from flashnext_gb10.mapped_table import MappedTable
from flashnext_gb10.vllm_nvme import gather_bytes

torch.manual_seed(818)
with tempfile.TemporaryDirectory() as directory:
    source = torch.randn(8192, 160, device="cpu").to(torch.float8_e4m3fn)
    storage = MappedTable(directory, source.numel())
    mapped = torch.frombuffer(storage.mapping, dtype=torch.float8_e4m3fn).reshape_as(source)
    mapped.view(torch.uint8).copy_(source.view(torch.uint8))
    storage.flush_rows(0, source.numel())
    for step in range(32):
        ids = torch.randint(0, 8192, (8, 16), device="cpu")
        ids[0, :3] = torch.tensor([8191, 0, 8191])
        rows = gather_bytes(mapped, ids, 0, 8192)
        reference = source.view(torch.uint8)[ids]
        assert torch.equal(rows, reference), f"CPU mismatch at step {step}"
        gpu = rows.pin_memory().to("cuda", non_blocking=True)
        assert torch.equal(gpu.cpu(), reference), f"GPU mismatch at step {step}"
    masked = gather_bytes(mapped, torch.tensor([[-1, 8192, 4]]), 0, 8192)
    assert not masked[0, :2].any()
    assert torch.equal(masked[0, 2], source.view(torch.uint8)[4])
    assert not mapped.is_pinned()
    print(json.dumps({"byte_exact_steps": 32, "agents": 8, "dtype": "FP8", "invalid_ids_zeroed": True,
                      "table_pinned": mapped.is_pinned(), "gpu": torch.cuda.get_device_name()}))
