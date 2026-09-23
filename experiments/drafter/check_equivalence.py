"""Check the transformers MTP block against vLLM's drafter on captured chunks.

Uses captures from FLASHNEXT_CAPTURE_MTP_DIR whose positions start at 0 (the
first chunk of a sequence, so attention sees the same context in both).
Reports hidden-state cosine similarity and argmax agreement of the draft
logits; kernel differences (NVFP4-era FP8 experts, fused ops) allow small drift.
"""
import argparse
from pathlib import Path

import torch

from mtp_torch import load_mtp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='directory with the extracted mtp.* weights and config')
    p.add_argument('--captures', required=True)
    p.add_argument('--limit', type=int, default=4)
    a = p.parse_args()
    block, embed, head = load_mtp(a.weights)
    block.eval()
    checked = 0
    for path in sorted(Path(a.captures).glob('*.pt')):
        record = torch.load(path)
        if record['positions'][0] != 0 or record['positions'][-1] == 0:  # later chunk, or warm-up dummy
            continue
        ids = record['ids'].long().cuda()[None]
        positions = record['positions'].long().cuda()[None]
        hidden = record['hidden'].cuda()[None]
        with torch.no_grad():
            sample, multi = block(hidden, embed[ids], positions)
        ref = record['sample_hidden'].cuda().float()
        ours = sample[0].float()
        cos = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
        top_ours = (ours.bfloat16() @ head.T).argmax(-1)
        top_ref = (ref.bfloat16() @ head.T).argmax(-1)
        multi_cos = torch.nn.functional.cosine_similarity(multi[0].float(), record['multi_hidden'].cuda().float(), dim=-1)
        print(f'{path.name}: T={ids.shape[1]} cos mean {cos.mean():.5f} min {cos.min():.4f} '
              f'multi cos {multi_cos.mean():.5f} argmax agree {(top_ours == top_ref).float().mean():.4f}')
        checked += 1
        if checked >= a.limit:
            break
    if not checked:
        raise SystemExit('no first-chunk captures found')


if __name__ == '__main__':
    main()
