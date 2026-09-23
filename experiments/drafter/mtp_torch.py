"""Differentiable Qwen4Exp MTP block built from the transformers modules.

Mirrors vLLM's Qwen4ExpMultiTokenPredictor (NVIDIA checkpoint):
  e = fc_embedding(norm(embed(token)))                 [T, H]
  h = fc_hidden(norm_4H(target_multi).view(T, 4, H))   [T, 4, H]
  x = h + e on every stream (pending combine, unit injection)
  x = decoder layer (QSA attention + MoE, hyperconnections)
  sample = hyper_connection_mixer(x) -> lm_head;  x is the next step's input.
Sequences up to the indexer budget select every visible token, so the indexer
is bypassed there (identical result, avoids its per-query Python loop).
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


def text_config(model_dir):
    raw = json.loads((Path(model_dir) / 'config.json').read_text())
    config = Qwen4ExpTextConfig(**raw.get('text_config', raw))
    config._attn_implementation = 'sdpa'
    return config


class MTPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        H, n, eps = config.hidden_size, config.hc_count, config.rms_norm_eps
        self.config = config
        self.pre_fc_norm_embedding = M.Qwen4ExpTextRMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = M.Qwen4ExpTextRMSNorm(H * n, eps=eps)
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)
        index = next(i for i, t in enumerate(config.layer_types)
                     if t != 'linear_attention' and i + 1 not in config.ple_layer_ids)
        self.layer = M.Qwen4ExpTextDecoderLayer(config, index)
        self.hyper_connection_mixer = M.Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.rotary = M.Qwen4ExpTextRotaryEmbedding(config)
        indexer = self.layer.self_attn.indexer
        original = indexer.forward

        def select_all_within_budget(hidden_states, position_embeddings, attention_mask, past_key_values):
            if attention_mask.shape[-1] <= indexer.token_budget:
                return torch.ones_like(attention_mask, dtype=torch.bool)
            return original(hidden_states, position_embeddings, attention_mask, past_key_values)

        indexer.forward = select_all_within_budget

    def forward(self, multi_hidden, token_embeds, positions):
        """multi_hidden [B, T, 4H], token_embeds [B, T, H], positions [B, T] -> (sample [B,T,H], multi [B,T,4H])."""
        B, T, _ = token_embeds.shape
        n, H = self.config.hc_count, self.config.hidden_size
        e = self.fc_embedding(self.pre_fc_norm_embedding(token_embeds))
        h = self.fc_hidden(self.pre_fc_norm_hidden(multi_hidden).view(B, T, n, H))
        x = (h + e.unsqueeze(-2)).flatten(-2)
        cos_sin = self.rotary(x, positions[None].expand(3, -1, -1))  # text: identical t/h/w positions
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None].expand(B, 1, T, T)
        x = self.layer(x, position_embeddings=cos_sin, attention_mask=mask)
        return self.hyper_connection_mixer(x), x


def _dequant_block_fp8(weight, scale_inv):
    rows, cols = weight.shape
    br, bc = -(-rows // scale_inv.shape[0]), -(-cols // scale_inv.shape[1])
    s = scale_inv.float().repeat_interleave(br, 0)[:rows].repeat_interleave(bc, 1)[:, :cols]
    return (weight.float() * s).to(torch.bfloat16)


def load_mtp(model_dir, device='cuda', dtype=torch.bfloat16):
    """Returns (block, embed_tokens weight, lm_head weight) with the checkpoint's mtp.* weights."""
    model_dir = Path(model_dir)
    config = text_config(model_dir)
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    names = [k for k in index if k.startswith('mtp.')]
    embed_name = next(k for k in index if k.endswith('embed_tokens.weight') and 'visual' not in k and not k.startswith('mtp'))
    head_name = next(k for k in index if k.endswith('lm_head.weight'))
    tensors = {}
    by_file = {}
    for k in names + [embed_name, head_name]:
        by_file.setdefault(index[k], []).append(k)
    for file, keys in by_file.items():
        with safe_open(str(model_dir / file), 'pt') as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    with torch.device('meta'):
        block = MTPBlock(config)
    state = {}
    prefix = 'mtp.layers.0.'
    experts = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {}}
    for k, v in tensors.items():
        if not k.startswith('mtp.') or k.endswith('_scale_inv'):
            continue
        scale = tensors.get(k + '_scale_inv')
        w = _dequant_block_fp8(v, scale) if v.dtype == torch.float8_e4m3fn else v.to(dtype)
        if k.startswith(prefix + 'mlp.experts.'):
            parts = k[len(prefix + 'mlp.experts.'):].split('.')
            experts[parts[1]][int(parts[0])] = w
            continue
        state[k[len('mtp.'):].replace('layers.0.', 'layer.')] = w
    E = config.num_experts
    state['layer.mlp.experts.gate_up_proj'] = torch.stack(
        [torch.cat([experts['gate_proj'][e], experts['up_proj'][e]]) for e in range(E)])
    state['layer.mlp.experts.down_proj'] = torch.stack([experts['down_proj'][e] for e in range(E)])
    block = block.to_empty(device=device)
    missing, unexpected = block.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.startswith('rotary.')]
    if missing or unexpected:
        raise RuntimeError(f'MTP weight mismatch: missing={missing} unexpected={unexpected}')
    block.rotary = M.Qwen4ExpTextRotaryEmbedding(config, device=device)
    block.to(dtype)
    embed = tensors[embed_name].to(device, dtype)
    head = tensors[head_name].to(device, dtype)
    return block, embed, head
