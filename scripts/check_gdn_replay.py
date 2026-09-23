"""Check the GDN replay decode (flashnext_gb10/gdn_replay.py) against vLLM's per-slot semantics.

A PyTorch reference implements what vLLM's fused MTP kernel does: start from
the slot of the accepted position, keep the state in FP32 across the step's
tokens and store it in BF16 after every token. Several requests run for
several steps with random acceptance, and the replay kernel's outputs and
current states are compared after each step. The run also crosses to a
non-fused path (materialize, then the reference) and back, and starts from a
freshly prefilled state. With --vllm, the reference is vLLM's CUDA op.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashnext_gb10.gdn_replay import materialize, replay_decode  # noqa: E402

H, HV, K, V = 16, 48, 128, 128


def reference(qkv, a, b, A_log, dt_bias, rows, cu, accepted, state, gate, norm_w, scale, eps, sigmoid):
    out = torch.zeros(qkv.shape[0], HV, V, dtype=torch.bfloat16, device=qkv.device)
    for r in range(rows.shape[0]):
        bos, eos = int(cu[r]), int(cu[r + 1])
        src = int(rows[r, int(accepted[r]) - 1])
        h = state[src].float()                           # [HV, V, K]
        for t in range(eos - bos):
            x = qkv[bos + t].float()
            q = x[:H * K].view(H, K).repeat_interleave(HV // H, 0)
            k = x[H * K:2 * H * K].view(H, K).repeat_interleave(HV // H, 0)
            v = x[2 * H * K:].view(HV, V)
            q = q * (torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale)
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            g = -A_log.exp() * F.softplus(a[bos + t].float() + dt_bias.float())
            decay, beta = g.exp(), torch.sigmoid(b[bos + t].float())
            h = h * decay[:, None, None]
            delta = (v - (h * k[:, None, :]).sum(-1)) * beta[:, None]
            h = h + delta[:, :, None] * k[:, None, :]
            o = (h * q[:, None, :]).sum(-1).bfloat16().float()
            rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
            z = gate[bos + t].float()
            act = torch.sigmoid(z) if sigmoid else F.silu(z)
            out[bos + t] = (o * rstd * norm_w.float() * act).bfloat16()
            state[int(rows[r, t])] = h.bfloat16()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', type=int, default=6)
    p.add_argument('--width', type=int, default=4)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--vllm', action='store_true', help="reference is vLLM's fused CUDA op")
    a_ = p.parse_args()
    torch.manual_seed(0)
    dev = 'cuda'
    B, W = a_.requests, a_.width
    slots = 1 + B * W + 3
    A_log = torch.rand(HV, device=dev) * 2 - 3
    dt_bias = torch.randn(HV, device=dev) * 0.5
    norm_w = torch.rand(V, device=dev, dtype=torch.bfloat16) + 0.5
    scale, eps = K ** -0.5, 1e-6
    perm = torch.randperm(slots - 1)[:B * W] + 1
    rows = perm.view(B, W).to(torch.int32).to(dev)
    init = (torch.randn(slots, HV, V, K, device=dev) * 0.05).bfloat16()
    init[0] = 0
    ref_state, rep_state = init.clone(), init.clone()
    accepted = torch.ones(B, dtype=torch.int32, device=dev)
    worst_out = worst_state = 0.0
    for step in range(a_.steps):
        lens = torch.full((B,), W, dtype=torch.int32)
        if step == 3:
            lens[0] = 1  # a request with no drafts this step
        cu = torch.zeros(B + 1, dtype=torch.int32)
        cu[1:] = lens.cumsum(0)
        cu = cu.to(dev)
        T = int(cu[-1])
        qkv = torch.randn(T, 2 * H * K + HV * V, device=dev).bfloat16()
        a = torch.randn(T, HV, device=dev).bfloat16()
        b = torch.randn(T, HV, device=dev).bfloat16()
        gate = torch.randn(T, HV, V, device=dev).bfloat16()
        sigmoid = step % 2 == 1
        if a_.vllm:
            from vllm import _custom_ops as ops
            ref_out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
            ops.fused_gdn_decode_post_conv_mtp(
                mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state_indices=rows, cu_seqlens=cu,
                num_accepted_tokens=accepted, state=ref_state, output_gate=gate, norm_weight=norm_w,
                out=ref_out, scale=scale, norm_eps=eps, output_gate_activation='sigmoid' if sigmoid else 'silu')
        else:
            ref_out = reference(qkv, a, b, A_log, dt_bias, rows.cpu(), cu.cpu(), accepted.cpu(), ref_state,
                                gate, norm_w, scale, eps, sigmoid)
        if step == 5:
            # A non-fused path this step: materialize, then vLLM's semantics on the replay state.
            materialize(rows, accepted, rep_state)
            rep_out = reference(qkv, a, b, A_log, dt_bias, rows.cpu(), cu.cpu(), accepted.cpu(), rep_state,
                                gate, norm_w, scale, eps, sigmoid)
        else:
            rep_out = torch.empty(T, HV, V, dtype=torch.bfloat16, device=dev)
            replay_decode(qkv, a, b, A_log, dt_bias, rows, cu, accepted, rep_state, gate, norm_w, rep_out,
                          H, scale, eps, sigmoid)
        torch.cuda.synchronize()
        err = ((rep_out.float() - ref_out.float()).abs().max() / ref_out.float().abs().max()).item()
        worst_out = max(worst_out, err)
        accepted = torch.tensor([int(torch.randint(1, int(n) + 1, ())) for n in lens], dtype=torch.int32,
                                device=dev)
        # Current state: reference at the accepted slot; replay after materializing into that slot.
        probe = rep_state.clone()
        materialize(rows, accepted, probe)
        s_err = 0.0
        for r in range(B):
            slot = int(rows[r, int(accepted[r]) - 1])
            ref_cur, rep_cur = ref_state[slot].float(), probe[slot].float()
            s_err = max(s_err, ((ref_cur - rep_cur).abs().max() / ref_cur.abs().max()).item())
        worst_state = max(worst_state, s_err)
        print(f'step {step}: accepted next {accepted.tolist()}  output rel err {err:.2e}  state rel err {s_err:.2e}')
        if step == 6:
            # A prefill overwrites request 1's first slot; its record must be ignored.
            fresh = (torch.randn(HV, V, K, device=dev) * 0.05).bfloat16()
            for st in (ref_state, rep_state):
                st[int(rows[1, 0])] = fresh
            accepted[1] = 1
    bad = worst_out > 2e-2 or worst_state > 2e-2
    print(f'worst output rel err {worst_out:.2e}, worst state rel err {worst_state:.2e}: {"FAIL" if bad else "OK"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
