"""Self-distil the MTP drafter on the served target's own hidden states.

Input: captures from FLASHNEXT_CAPTURE_MTP_DIR (drafter inputs over prefill
chunks of replayed target generations). Drafter row i receives the target's
multi-stream hidden at position i and token i+1 and should predict the target's
distribution at position i+1, which is lm_head(target_mixer(hidden[i+1])).
Draft step k (unrolled, k = 1..3) consumes step k-1's multi stream and token
i+k. Routed experts, embeddings and lm_head stay frozen.

Metric: chained greedy acceptance, the fraction of rows whose first j draft
argmaxes (restricted to the draft vocabulary) all match the target argmax,
averaged into an expected accepted length 1 + sum_j P(first j accepted).
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen4_exp import modeling_qwen4_exp as M

from mtp_torch import load_mtp, text_config


def fp8_ste(x):
    """The served FP8 KV cache (unit scale) with a straight-through gradient."""
    return x + (x.to(torch.float8_e4m3fn).to(x.dtype) - x).detach()


class DraftContext:
    """Serving-equivalent attention for unrolled draft steps.

    When the server drafts, step 1 runs on verified positions from the
    target's hidden states and writes the drafter's KV cache there. Step k > 1
    at position p attends to that step-1 KV at positions < p plus its own KV
    at p. Step 1's keys and values are kept here and reused by later steps.
    """
    step = 1
    kv_fp8 = False
    keys = values = None


def draft_attention(module, q, k, v, mask, scaling=None, **kw):
    if DraftContext.kv_fp8:
        k, v = fp8_ste(k), fp8_ste(v)
    if DraftContext.step == 1:
        DraftContext.keys, DraftContext.values = k, v
        return sdpa_attention_forward(module, q, k, v, mask, scaling=scaling, **kw)
    groups = q.shape[1] // k.shape[1]
    k1 = DraftContext.keys.repeat_interleave(groups, 1)
    v1 = DraftContext.values.repeat_interleave(groups, 1)
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    rows, context = q.shape[2], k1.shape[2]
    offset = DraftContext.step - 1  # row i sits at position i + offset
    scores = (q.float() @ k1.float().transpose(-1, -2)) * scaling
    visible = torch.arange(context, device=q.device)[None] < (torch.arange(rows, device=q.device)[:, None] + offset)
    scores = scores.masked_fill(~visible, -math.inf)
    own = (q.float() * k.float()).sum(-1, keepdim=True) * scaling
    probs = torch.cat([scores, own], -1).softmax(-1)
    out = probs[..., :context] @ v1.float() + probs[..., context:] * v.float()
    return out.to(q.dtype).transpose(1, 2).contiguous(), None


AttentionInterface.register('draft', draft_attention)


def quantize_fp8_block(w, block=128):
    """[N, K] -> (e4m3 [N, K], bf16 scale_inv [ceil(N/128), ceil(K/128)]), the checkpoint's expert format."""
    N, K = w.shape
    pn, pk = (-N) % block, (-K) % block
    blocks = F.pad(w.float(), (0, pk, 0, pn)).view((N + pn) // block, block, (K + pk) // block, block)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12).to(torch.bfloat16)
    q = (blocks / scale.float()[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(N + pn, K + pk)[:N, :K].contiguous(), scale


def dequantize_fp8_block(q, scale_inv, dtype, block=128):
    N, K = q.shape
    s = scale_inv.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (q.float() * s).to(dtype)


def load_sequences(capture_dirs):
    """Join chunk captures into sequences (a sequence restarts at position 0)."""
    sequences, current = [], None
    paths = [path for d in capture_dirs for path in sorted(Path(d).glob('*.pt'))]
    for path in paths:
        record = torch.load(path, mmap=True)
        if record['positions'].shape[0] > 1 and record['positions'][-1] == 0:
            continue  # startup profiling / warm-up forward on dummy tokens
        if record['positions'][0] == 0 or current is None:
            if current:
                sequences.append(current)
            current = dict(ids=[], positions=[], hidden=[])
        elif record['positions'][0] != current['positions'][-1][-1] + 1:
            raise ValueError(f'{path.name}: chunk does not continue the previous sequence')
        for key in current:
            current[key].append(record[key].clone())
    if current:
        sequences.append(current)
    return [{k: torch.cat(v) for k, v in s.items()} for s in sequences]


def windows(sequences, length):
    out = []
    for s in sequences:
        T = s['ids'].shape[0]
        for start in range(0, T - 4, length):
            end = min(T, start + length)
            if end - start >= 64:
                out.append({k: v[start:end] for k, v in s.items()})
    return out


class TargetMixer(torch.nn.Module):
    def __init__(self, config, path, device):
        super().__init__()
        self.mixer = M.Qwen4ExpTextGatedResidual(config, use_combine=False)
        state = {k.split('hyper_connection_mixer.')[1]: v for k, v in load_file(path).items()}
        self.mixer.load_state_dict(state)
        self.to(device, torch.bfloat16).requires_grad_(False)

    def forward(self, multi):
        return self.mixer(multi)


def unrolled(block, embed, window, steps):
    """Yields (step, draft sample hidden [rows, H], target row offset) for k = 1..steps."""
    ids = window['ids'].long().cuda()
    positions = window['positions'].long().cuda()
    multi = window['hidden'].cuda()
    T = ids.shape[0]
    for k in range(1, steps + 1):
        DraftContext.step = k
        rows = T - k  # row i predicts target position i + k
        # Step k at row i drafts from position i + k - 1 with token i + k - 1 (teacher-forced).
        sample, multi = block(multi[:rows][None], embed[ids[k - 1:k - 1 + rows]][None],
                              positions[k - 1:k - 1 + rows][None])
        multi = multi[0]
        yield k, sample[0], multi


def kl_and_accept(sample, target_logits, head, vocab_mask, chunk=512):
    loss = 0.0
    hits = []
    for s in range(0, sample.shape[0], chunk):
        q = (sample[s:s + chunk].to(head.dtype) @ head.T).float()
        p = target_logits[s:s + chunk]
        loss = loss + F.kl_div(F.log_softmax(q, -1), F.log_softmax(p, -1), log_target=True, reduction='sum')
        draft = q.masked_fill(~vocab_mask, -math.inf).argmax(-1) if vocab_mask is not None else q.argmax(-1)
        hits.append(draft == p.argmax(-1))
    return loss / sample.shape[0], torch.cat(hits)


def target_logits(mixer, head, hidden, start, rows, chunk=512):
    out = []
    with torch.no_grad():
        for s in range(start, start + rows, chunk):
            e = min(start + rows, s + chunk)
            out.append((mixer(hidden[s:e].cuda()) @ head.T).float())
    return torch.cat(out)


def evaluate(block, embed, head, mixer, data, steps, vocab_mask):
    block.eval()
    chained = torch.zeros(steps)
    count = 0
    with torch.no_grad():
        for window in data:
            accepted = None
            for k, sample, _ in unrolled(block, embed, window, steps):
                rows = sample.shape[0]
                logits = target_logits(mixer, head, window['hidden'], k, rows)
                _, hits = kl_and_accept(sample, logits, head, vocab_mask)
                n = window['ids'].shape[0] - steps
                hits = hits[:n]
                accepted = hits if accepted is None else accepted & hits
                chained[k - 1] += accepted.float().sum().item()
            count += window['ids'].shape[0] - steps
    block.train()
    rates = (chained / count).tolist()
    return dict(chained_acceptance=rates, expected_accepted_length=1 + sum(rates))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--captures', required=True, nargs='+', help='one or more capture directories')
    p.add_argument('--draft-vocab', help='JSON list of draft token IDs (the served draft vocabulary)')
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--step-weights', default='1.0,0.7,0.5')
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--holdout', type=float, default=0.1)
    p.add_argument('--kv-dtype', choices=['bf16', 'fp8'], default='fp8',
                   help='fp8 matches a server run with FLASHNEXT_KV_DTYPE=fp8')
    p.add_argument('--train-experts', action='store_true',
                   help='also train the routed experts; exported re-quantized to the checkpoint FP8 block format')
    p.add_argument('--expert-lr', type=float, default=None, help='defaults to lr / 4')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    config = text_config(a.weights)
    block, embed, head = load_mtp(a.weights)
    # FP32 master weights: an AdamW step (about lr) is below a BF16 ulp for most
    # weights, so BF16 parameters would silently drop most updates.
    block.float()
    block.config._attn_implementation = 'draft'
    DraftContext.kv_fp8 = a.kv_dtype == 'fp8'
    embed.requires_grad_(False)
    head.requires_grad_(False)
    mixer = TargetMixer(config, str(Path(a.weights) / 'target_mixer.safetensors'), 'cuda')
    vocab_mask = None
    if a.draft_vocab:
        vocab = json.loads(Path(a.draft_vocab).read_text())
        vocab = vocab if isinstance(vocab, list) else vocab['token_ids']
        vocab_mask = torch.zeros(head.shape[0], dtype=torch.bool, device='cuda')
        vocab_mask[torch.tensor(vocab, device='cuda')] = True
    sequences = load_sequences(a.captures)
    random.Random(0).shuffle(sequences)
    split = max(1, int(len(sequences) * a.holdout))
    held, train = windows(sequences[:split], a.window), windows(sequences[split:], a.window)
    print(f'{len(sequences)} sequences: {len(train)} train windows, {len(held)} held-out', flush=True)
    for name, param in block.named_parameters():
        param.requires_grad_(a.train_experts or 'mlp.experts.' not in name)
    trainable = [q for q in block.parameters() if q.requires_grad]
    expert_params = [q for n, q in block.named_parameters() if q.requires_grad and 'mlp.experts.' in n]
    other_params = [q for n, q in block.named_parameters() if q.requires_grad and 'mlp.experts.' not in n]
    print(f'trainable parameters: {sum(q.numel() for q in trainable) / 1e6:.1f}M', flush=True)
    report = dict(before=evaluate(block, embed, head, mixer, held, a.steps, vocab_mask))
    print('before', report['before'], flush=True)
    groups = [dict(params=other_params, lr=a.lr)]
    if expert_params:
        groups.append(dict(params=expert_params, lr=a.expert_lr or a.lr / 4))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    total = a.epochs * len(train)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(1.0, (i + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total)))
    weights = [float(w) for w in a.step_weights.split(',')]
    step = 0
    for epoch in range(a.epochs):
        random.Random(epoch).shuffle(train)
        for window in train:
            loss = 0.0
            for k, sample, _ in unrolled(block, embed, window, a.steps):
                logits = target_logits(mixer, head, window['hidden'], k, sample.shape[0])
                kl, _ = kl_and_accept(sample, logits, head, None)
                loss = loss + weights[k - 1] * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 20 == 0:
                print(f'step {step}/{total} loss {loss.item():.4f}', flush=True)
    report['after'] = evaluate(block, embed, head, mixer, held, a.steps, vocab_mask)
    print('after', report['after'], flush=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    state = {f'mtp.{k.replace("layer.", "layers.0.", 1)}': v.detach().to(torch.bfloat16).cpu()
             for k, v in block.state_dict().items()
             if 'mlp.experts.' not in k and not k.startswith('rotary.')}
    if a.train_experts:
        experts = block.layer.mlp.experts
        with torch.no_grad():
            half = experts.gate_up_proj.shape[1] // 2
            for e in range(experts.gate_up_proj.shape[0]):
                for proj, w in (('gate_proj', experts.gate_up_proj[e, :half]), ('up_proj', experts.gate_up_proj[e, half:]),
                                ('down_proj', experts.down_proj[e])):
                    q, scale_inv = quantize_fp8_block(w)
                    state[f'mtp.layers.0.mlp.experts.{e}.{proj}.weight'] = q.cpu()
                    state[f'mtp.layers.0.mlp.experts.{e}.{proj}.weight_scale_inv'] = scale_inv.cpu()
                    # Evaluate what will be served: the re-quantized weights.
                    w.copy_(dequantize_fp8_block(q, scale_inv, w.dtype))
        report['after_fp8_experts'] = evaluate(block, embed, head, mixer, held, a.steps, vocab_mask)
        print('after (FP8 experts)', report['after_fp8_experts'], flush=True)
    torch.save(state, out / 'mtp_trained.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
