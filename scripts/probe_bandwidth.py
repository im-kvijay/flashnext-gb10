"""Measure achievable GB10 unified-memory bandwidth with read-dominant kernels."""
import json
import torch

def timed(fn, iters=20):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3

out = {"device": torch.cuda.get_device_name(), "clock_note": "see nvidia-smi"}
n_bytes = 2 * 1024**3
x = torch.empty(n_bytes // 2, dtype=torch.bfloat16, device="cuda").normal_()
y = torch.empty_like(x)
t = timed(lambda: x.sum(dtype=torch.float32))
out["read_sum_GBps"] = n_bytes / t / 1e9
t = timed(lambda: y.copy_(x))
out["copy_rw_GBps"] = 2 * n_bytes / t / 1e9
# GEMV-shaped read: BF16 weight [N, 2560] times 24 token activations, like decode projections.
rows = x.numel() // 2560
w = x[: rows * 2560].view(rows, 2560)
w_bytes = w.numel() * 2
for m in (8, 24):
    a = torch.randn(m, 2560, dtype=torch.bfloat16, device="cuda")
    t = timed(lambda: a @ w.T)
    out[f"bf16_gemm_m{m}_GBps"] = w_bytes / t / 1e9
print(json.dumps(out, indent=2))
