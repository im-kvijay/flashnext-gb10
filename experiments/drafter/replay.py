"""Replay generated sequences one at a time through a capturing server.

The server runs with FLASHNEXT_EAGER=1, MTP enabled and
FLASHNEXT_CAPTURE_MTP_DIR set; each prompt-only request (prompt plus recorded
continuation, max_tokens=1) prefills in chunks, and every chunk's drafter
inputs and outputs are saved. Requests are sequential so a chunk never mixes
sequences.
"""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--generations', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output', help='summary JSON')
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.generations).read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[:a.limit]
    total = 0
    for row in rows:
        ids = row['prompt_token_ids'] + row['output_token_ids']
        payload = json.dumps({'model': 'flashnext', 'prompt': ids, 'max_tokens': 1, 'temperature': 0}).encode()
        request = urllib.request.Request(a.url + '/v1/completions', data=payload,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3600) as response:
            json.load(response)
        total += len(ids)
        print(f"{row['id']}: {len(ids)} tokens (total {total})", flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps({'sequences': len(rows), 'tokens': total}))


if __name__ == '__main__':
    main()
