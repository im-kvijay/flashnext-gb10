"""Localize torch-vs-vLLM MTP drift: cosine by position bucket and per stage."""
import argparse
from pathlib import Path

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from mtp_torch import load_mtp


def fp8(x):
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def fp8_group(x, group=128):
    """Dynamic per-token, per-128-column FP8 activation quantization (vLLM block FP8)."""
    shape = x.shape
    g = x.float().reshape(*shape[:-1], -1, group)
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    return ((g / scale).to(torch.float8_e4m3fn).float() * scale).reshape(shape).to(x.dtype)


def kv_fp8_attention(module, q, k, v, mask, **kw):
    return sdpa_attention_forward(module, q, fp8(k), fp8(v), mask, **kw)


def qkv_fp8_attention(module, q, k, v, mask, **kw):
    return sdpa_attention_forward(module, fp8(q), fp8(k), fp8(v), mask, **kw)


def fp8_scaled(x):
    scale = x.float().abs().amax().clamp(min=1e-12) / 448.0
    return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


def fp8_head_scaled(x):
    """Per-KV-head scale (x is [B, heads, T, D])."""
    scale = x.float().abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-12) / 448.0
    return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


def kv_fp8_scaled_attention(module, q, k, v, mask, **kw):
    print(f'    k amax {k.abs().amax():.3f} rms {k.float().pow(2).mean().sqrt():.3f} '
          f'v amax {v.abs().amax():.4f} rms {v.float().pow(2).mean().sqrt():.4f}', flush=True)
    return sdpa_attention_forward(module, q, fp8_scaled(k), fp8_scaled(v), mask, **kw)


def kv_fp8_head_attention(module, q, k, v, mask, **kw):
    return sdpa_attention_forward(module, q, fp8_head_scaled(k), fp8_head_scaled(v), mask, **kw)


AttentionInterface.register('kv_fp8', kv_fp8_attention)
AttentionInterface.register('kv_fp8_scaled', kv_fp8_scaled_attention)
AttentionInterface.register('kv_fp8_head', kv_fp8_head_attention)
AttentionInterface.register('qkv_fp8', qkv_fp8_attention)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--captures', required=True)
    p.add_argument('--limit', type=int, default=2)
    a = p.parse_args()
    block, embed, head = load_mtp(a.weights)
    block.eval()
    experts = block.layer.mlp.experts
    plain_experts = experts.forward

    def act_fp8_experts(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for e in top_k_index.unique().tolist():
            pos, slot = (top_k_index == e).nonzero(as_tuple=True)
            gate, up = torch.nn.functional.linear(fp8_group(hidden_states[pos]), experts.gate_up_proj[e]).chunk(2, -1)
            h = torch.nn.functional.linear(fp8_group(experts.act_fn(gate) * up), experts.down_proj[e])
            final.index_add_(0, pos, (h * top_k_weights[pos, slot, None]).to(final.dtype))
        return final

    variants = {
        'bf16': ('sdpa', plain_experts),
        'kv_fp8': ('kv_fp8', plain_experts),
        'kv_fp8_scaled': ('kv_fp8_scaled', plain_experts),
        'kv_fp8_head': ('kv_fp8_head', plain_experts),
    }
    checked = 0
    for path in sorted(Path(a.captures).glob('*.pt')):
        record = torch.load(path)
        if record['positions'][0] != 0 or record['positions'][-1] == 0:
            continue
        ids = record['ids'].long().cuda()[None]
        positions = record['positions'].long().cuda()[None]
        hidden = record['hidden'].cuda()[None]
        ref = record['sample_hidden'].cuda().float()
        top_ref = (ref.bfloat16() @ head.T).argmax(-1)
        for name, (attention, expert_forward) in variants.items():
            block.config._attn_implementation = attention
            experts.forward = expert_forward
            with torch.no_grad():
                sample, multi = block(hidden, embed[ids], positions)
            cos = torch.nn.functional.cosine_similarity(sample[0].float(), ref, dim=-1)
            T = cos.shape[0]
            buckets = [cos[i:i + 64].mean().item() for i in range(0, T, 64)]
            agree = ((sample[0] @ head.T).argmax(-1) == top_ref).float().mean().item()
            print(f'{path.name} T {T} {name:20s} cos {cos.mean():.4f} argmax {agree:.4f} by-64',
                  ' '.join(f'{b:.3f}' for b in buckets), flush=True)
        checked += 1
        if checked >= a.limit:
            break


if __name__ == '__main__':
    main()
