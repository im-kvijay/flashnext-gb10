"""Memory-bandwidth ceiling for concurrent Flash Next decode on one device.

Every decode step must read each dense weight once, each distinct routed expert
used by that step's target tokens once, and the recurrent GDN state. Drafting
adds draft-head reads. Bytes come from the checkpoint's safetensors headers;
bandwidth should be measured on the target host (see scripts/probe_bandwidth.py).
Output is an upper bound on output tokens/s, not a throughput measurement.
"""
import argparse
import glob
import json
import os
import struct
from collections import Counter


def categorize(name):
    if 'visual' in name or 'vision' in name:
        return 'vision'
    if name.startswith('mtp'):
        return 'mtp_experts' if '.experts.' in name else 'mtp_dense'
    if 'ngram' in name:
        return 'ple_table'
    if '.experts.' in name:
        return 'experts'
    if 'embed_tokens' in name:
        return 'embed_tokens'
    return 'dense'


def checkpoint_bytes(path):
    totals = Counter()
    for shard in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        with open(shard, 'rb') as stream:
            header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
        for name, tensor in header.items():
            if name != '__metadata__':
                start, end = tensor['data_offsets']
                totals[categorize(name)] += end - start
            if name.endswith('lm_head.weight'):
                totals['lm_head'] += end - start
    return totals


def unique_experts(tokens, experts, top_k):
    """Expected distinct experts under independent uniform routing (upper bound)."""
    return experts * (1 - (1 - top_k / experts) ** tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--bandwidth-gbps', type=float, required=True,
                   help='measured achievable read bandwidth')
    p.add_argument('--agents', type=int, default=8)
    p.add_argument('--draft-vocab', type=int, default=None,
                   help='reduced draft projection rows; default is the full vocabulary')
    p.add_argument('--expert-fraction', type=float, default=None,
                   help='observed distinct experts / uniform-routing estimate')
    p.add_argument('--output')
    a = p.parse_args()
    config = json.load(open(os.path.join(a.checkpoint, 'config.json')))
    text = config.get('text_config', config)
    layers, experts, top_k = text['num_hidden_layers'], text['num_experts'], text['num_experts_per_tok']
    vocab = text['vocab_size']
    b = checkpoint_bytes(a.checkpoint)
    per_expert = b['experts'] / (layers * experts)
    mtp_per_expert = b['mtp_experts'] / experts
    gdn_layers = sum(t == 'linear_attention' for t in text['layer_types'])
    state_bytes = (gdn_layers * text['linear_num_value_heads'] * text['linear_key_head_dim']
                   * text['linear_value_head_dim'] * 4)
    draft_head = b['lm_head'] * (a.draft_vocab or vocab) / vocab
    bw = a.bandwidth_gbps * 1e9
    rows = []
    for k in range(0, 8):
        tokens = a.agents * (k + 1)
        target_experts = unique_experts(tokens, experts, top_k)
        if a.expert_fraction is not None:
            target_experts *= a.expert_fraction
        draft_step = (b['mtp_dense'] + draft_head
                      + mtp_per_expert * unique_experts(a.agents, experts, top_k))
        fixed = b['dense'] + 2 * a.agents * state_bytes + k * draft_step
        routed = layers * target_experts * per_expert
        for label, total in (('experts_free', fixed), ('with_experts', fixed + routed)):
            step = total / bw
            rows.append(dict(draft_tokens=k, bound=label, bytes_per_step_gb=total / 1e9,
                             step_ms_floor=step * 1e3,
                             ceiling_tps_perfect_acceptance=a.agents * (k + 1) / step,
                             acceptance_needed_for_400=400 * step / a.agents))
    report = dict(scope='bandwidth ceiling; not a throughput measurement',
                  bandwidth_gbps=a.bandwidth_gbps, agents=a.agents,
                  bytes_gb={k: v / 1e9 for k, v in sorted(b.items())},
                  bytes_per_routed_expert_mb=per_expert / 1e6,
                  gdn_state_mb_per_agent=state_bytes / 1e6, rows=rows)
    if a.output:
        with open(a.output, 'x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print('k  bound         GB/step  floor_ms  max_tps(all accepted)  mean_accept_needed_for_400')
    for r in rows:
        print(f"{r['draft_tokens']}  {r['bound']:<12} {r['bytes_per_step_gb']:8.2f} {r['step_ms_floor']:8.1f}"
              f" {r['ceiling_tps_perfect_acceptance']:12.1f} {r['acceptance_needed_for_400']:16.2f}")


if __name__ == '__main__':
    main()
