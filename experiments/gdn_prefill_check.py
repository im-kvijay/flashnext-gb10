"""GDN chunked-prefill kernels versus a naive fp32 recurrence on this GPU.

Shapes follow Qwen3.8-Flash-Next (16 key heads, 48 value heads, dim 128).
Runs vLLM's FlashInfer path and its bundled FLA Triton kernel twice each on
the same inputs (determinism) and compares both with the reference.
"""
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fi_chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as fla_chunk

torch.manual_seed(0)
T, HK, HV, D = 2048, 16, 48, 128
dev = 'cuda'
q = torch.nn.functional.normalize(torch.randn(1, T, HK, D, device=dev), dim=-1).bfloat16()
k = torch.nn.functional.normalize(torch.randn(1, T, HK, D, device=dev), dim=-1).bfloat16()
v = torch.randn(1, T, HV, D, device=dev).bfloat16()
g = (-torch.rand(1, T, HV, device=dev) * 0.5).float()          # log decay
beta = torch.rand(1, T, HV, device=dev).bfloat16()
state0 = torch.zeros(1, HV, D, D, device=dev)
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)


def reference():
    rep = HV // HK
    qf = q[0].float().repeat_interleave(rep, 1)
    kf = k[0].float().repeat_interleave(rep, 1)
    vf, gf, bf = v[0].float(), g[0].float(), beta[0].float()
    S = torch.zeros(HV, D, D, device=dev)  # [h, k, v]
    out = torch.empty(T, HV, D, device=dev)
    scale = D ** -0.5
    for t in range(T):
        S = S * gf[t].exp()[:, None, None]
        kv = torch.einsum('hk,hkv->hv', kf[t], S)
        u = bf[t][:, None] * (vf[t] - kv)
        S = S + torch.einsum('hk,hv->hkv', kf[t], u)
        out[t] = torch.einsum('hk,hkv->hv', qf[t] * scale, S)
    return out


def run_fi():
    o, s = fi_chunk_gated_delta_rule(q, k, v, g, beta, state0.clone(), True, cu_seqlens=cu,
                                     use_qk_l2norm_in_kernel=False)
    return o[0].float()


def run_fla():
    o, s = fla_chunk(q=q, k=k, v=v, g=g, beta=beta, initial_state=state0.clone(), output_final_state=True,
                     cu_seqlens=cu.long(), use_qk_l2norm_in_kernel=False)
    return o[0].float()


ref = reference()


def report(name, a, b):
    rel = lambda x: ((x - ref).norm(dim=(-1, -2)) / ref.norm(dim=(-1, -2)))
    ra = rel(a)
    print(f'{name}: vs ref mean {ra.mean():.2e} max {ra.max():.2e} worst token {ra.argmax().item()} '
          f'| run-to-run max abs diff {(a - b).abs().max():.2e} bitwise {torch.equal(a, b)}')
    for lo in (0, 64, 512, 1024, 1536):
        print(f'   tokens {lo}-{lo + 63}: rel {ra[lo:lo + 64].mean():.2e}')


for name, fn in (('flashinfer', run_fi), ('fla-triton', run_fla)):
    try:
        report(name, fn(), fn())
    except Exception as e:  # noqa: BLE001
        print(name, 'failed:', repr(e)[:300])
