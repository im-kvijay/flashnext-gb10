"""Streaming full-model reference for Qwen3.8-Flash-Next (transformers modules).

Evaluates the whole text model one decoder layer at a time: NVFP4 experts
are dequantized to BF16 and evaluated with FP32 activations, all other
weights in FP32, PLE rows read directly from the checkpoint's FP8 table and
scaled by its global scale. Sequences stay within the QSA budget (2048), where
sparse attention equals dense causal attention. Reports the realized-token
NLL of the same spans that bench/determinism.py scores, and compares
per-position log-probabilities with a served record.

Usage: model_reference.py --model DIR --record determinism.json
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


class Checkpoint:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.index = json.loads((self.dir / 'model.safetensors.index.json').read_text())['weight_map']
        self.handles = {}

    def handle(self, name):
        file = self.index[name]
        if file not in self.handles:
            self.handles[file] = safe_open(str(self.dir / file), 'pt')
        return self.handles[file]

    def get(self, name):
        return self.handle(name).get_tensor(name)

    def keys(self, prefix):
        return [k for k in self.index if k.startswith(prefix)]


def dequant_nvfp4(packed, scale, scale2):
    packed = packed.cuda()
    lut = E2M1.cuda()
    values = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], -1).flatten(-2)
    return (values * scale.cuda().float().repeat_interleave(16, -1) * scale2.cuda().float()).to(torch.bfloat16)


class RowFetcher(torch.nn.Module):
    """PLE rows from the FP8 table shards, times the table's global scale."""

    def __init__(self, ckpt, prefix):
        super().__init__()
        self.ckpt, self.prefix = ckpt, prefix
        self.rows = ckpt.handle(prefix + 'shard_0.weight').get_slice(prefix + 'shard_0.weight').get_shape()[0]
        self.scale = ckpt.get(prefix + 'weight_scale').float().item()
        # transformers reads ngram_embedding.weight.device to place the IDs.
        self.register_buffer('weight', torch.empty(0), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1).cpu()
        unique, inverse = flat.unique(return_inverse=True)
        out = torch.empty(len(unique), 160)
        for i, row in enumerate(unique.tolist()):
            shard, local = divmod(row, self.rows)
            name = f'{self.prefix}shard_{shard}.weight'
            out[i] = self.ckpt.handle(name).get_slice(name)[local:local + 1][0].float()
        return (out[inverse] * self.scale).reshape(*ids.shape, 160).to(ids.device)


def experts_forward_fp32(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    for e in top_k_index.unique().tolist():
        pos, slot = (top_k_index == e).nonzero(as_tuple=True)
        x = hidden_states[pos]
        gate, up = torch.nn.functional.linear(x, self.gate_up_proj[e].float()).chunk(2, dim=-1)
        h = torch.nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e].float())
        final.index_add_(0, pos, h * top_k_weights[pos, slot, None])
    return final


def build_layer(config, ckpt, index):
    prefix = f'model.language_model.layers.{index}.'
    with torch.device('meta'):
        layer = M.Qwen4ExpTextDecoderLayer(config, index)
    if layer.ple is not None:
        # Never materialize the 320M-row table: rows are read from the checkpoint on demand.
        layer.ple.ple_embedding.ngram_embedding = RowFetcher(ckpt, prefix + 'ple.ple_embedding.ngram_embedding.')
    layer = layer.to_empty(device='cuda')
    state = {}
    experts = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {}}
    for name in ckpt.keys(prefix):
        local = name[len(prefix):]
        if '.ngram_embedding.' in local:
            continue
        if '.experts.' in local:
            if local.endswith('.weight'):
                e, proj = local.split('.')[2:4]
                experts[proj][int(e)] = dequant_nvfp4(ckpt.get(name), ckpt.get(name + '_scale'), ckpt.get(name + '_scale_2'))
            continue
        tensor = ckpt.get(name)
        state[local] = tensor if tensor.dtype == torch.int64 else tensor.float()
    E = config.num_experts
    layer.mlp.experts.gate_up_proj = torch.nn.Parameter(
        torch.stack([torch.cat([experts['gate_proj'][e], experts['up_proj'][e]]) for e in range(E)]), requires_grad=False)
    layer.mlp.experts.down_proj = torch.nn.Parameter(torch.stack([experts['down_proj'][e] for e in range(E)]),
                                                     requires_grad=False)
    missing, unexpected = layer.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.startswith('mlp.experts.') and 'ngram_embedding' not in m]
    if missing or unexpected:
        raise RuntimeError(f'layer {index}: missing {missing} unexpected {unexpected}')
    layer.mlp.experts.forward = experts_forward_fp32.__get__(layer.mlp.experts)
    attention = getattr(layer, 'self_attn', None)
    if attention is not None:
        attention.indexer.forward = lambda h, pe, mask, cache: torch.ones_like(mask, dtype=torch.bool)
    return layer.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--record', required=True, help='bench/determinism.py output with codebase-0 ids and rows')
    p.add_argument('--output')
    a = p.parse_args()
    torch.set_grad_enabled(False)
    raw = json.loads((Path(a.model) / 'config.json').read_text())
    config = Qwen4ExpTextConfig(**raw.get('text_config', raw))
    config._attn_implementation = 'eager'
    ckpt = Checkpoint(a.model)
    served = next(r for r in json.loads(Path(a.record).read_text())['records'] if r['name'] == 'codebase-0')
    ids = served['ids']
    sequences = {'full-2048': ids[:2048]}
    for offset in (512, 1024, 1536):
        sequences[f'short@{offset}'] = ids[:32] + ids[offset - 128:offset + 256]
    lm = 'model.language_model.'
    embed = ckpt.get(lm + 'embed_tokens.weight').float().cuda()
    states, masks, rotary_inputs, token_ids = {}, {}, {}, {}
    rotary = M.Qwen4ExpTextRotaryEmbedding(config, device='cuda')
    for name, seq in sequences.items():
        t = torch.tensor(seq, device='cuda')[None]
        T = t.shape[1]
        token_ids[name] = t
        states[name] = embed[t].repeat(1, 1, config.hc_count)
        pos = torch.arange(T, device='cuda')[None, None].expand(3, 1, -1)
        rotary_inputs[name] = rotary(states[name], pos)
        masks[name] = torch.zeros(1, 1, T, T, device='cuda').masked_fill(
            ~torch.ones(T, T, dtype=torch.bool, device='cuda').tril(), float('-inf'))
    del embed
    for index in range(config.num_hidden_layers):
        start = time.time()
        layer = build_layer(config, ckpt, index)
        built = time.time() - start
        for name in sequences:
            states[name] = layer(states[name], position_embeddings=rotary_inputs[name], attention_mask=masks[name],
                                 conv_mask=None, past_key_values=None, ple_input_ids=token_ids[name])
        del layer
        gc.collect()  # the patched experts.forward is a bound method stored on its own module: a cycle
        torch.cuda.empty_cache()
        print(f'layer {index}: build {built:.0f}s forward {time.time() - start - built:.0f}s', flush=True)
    mixer = M.Qwen4ExpTextGatedResidual(config, use_combine=False).cuda()
    mixer.load_state_dict({k.split('hyper_connection_mixer.')[1]: ckpt.get(k).float()
                           for k in ckpt.keys(lm + 'hyper_connection_mixer.')})
    head = ckpt.get('lm_head.weight').float().cuda()
    report = {}
    logprobs = {}
    for name, seq in sequences.items():
        logits = mixer(states[name])[0] @ head.T
        lp = logits.log_softmax(-1)
        t = token_ids[name][0]
        realized = lp[:-1].gather(1, t[1:, None])[:, 0]  # realized[i] = log p(token i+1)
        logprobs[name] = realized.cpu()
    full = logprobs['full-2048']
    for offset in (512, 1024, 1536):
        full_span = -full[offset - 1:offset + 255].mean().item()
        short = logprobs[f'short@{offset}']
        short_span = -short[-256:].mean().item()
        report[f'span@{offset} nll full/short'] = [full_span, short_span]
    served_rows = torch.tensor([r[0] for r in served['rows']])  # positions 256.. of the served sequence
    n = 2048 - 256
    ref = full[255:255 + n]
    diff = (ref - served_rows[:n]).abs()
    report['reference mean nll 256..2047'] = -ref.mean().item()
    report['served mean nll 256..2047'] = -served_rows[:n].mean().item()
    report['served vs reference |dlogp| mean'] = diff.mean().item()
    report['served vs reference frac |dlogp|>0.5'] = (diff > 0.5).float().mean().item()
    print(json.dumps(report, indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(dict(report=report, reference_logprobs=full.tolist())))


if __name__ == '__main__':
    main()
