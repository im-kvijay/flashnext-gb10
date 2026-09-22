"""Download and freeze the task-suite samples used by bench/suite.py.

Needs pyarrow and huggingface_hub (use any Python; the serving runtime does not
need pyarrow). Samples are fixed by seed so every configuration is graded on
identical items.
"""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SOURCES = {
    'humaneval': ('openai/openai_humaneval', 'openai_humaneval/test-00000-of-00001.parquet', None, None),
    'gsm8k': ('openai/gsm8k', 'main/test-00000-of-00001.parquet', 100, None),
    'mmlu_pro': ('TIGER-Lab/MMLU-Pro', 'data/test-00000-of-00001.parquet', 140, 'category'),
}

p = argparse.ArgumentParser()
p.add_argument('--output', default='data/eval')
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
lcb = hf_hub_download('livecodebench/code_generation_lite', 'test6.jsonl', repo_type='dataset')
(out / 'livecodebench_test6.jsonl').write_bytes(Path(lcb).read_bytes())
print('livecodebench_test6', sum(1 for _ in open(lcb)))
for name, (repo, path, n, key) in SOURCES.items():
    rows = pq.read_table(hf_hub_download(repo, path, repo_type='dataset')).to_pylist()
    if n:
        random.Random(0).shuffle(rows)
        if key:
            groups = {}
            for row in rows:
                groups.setdefault(row[key], []).append(row)
            rows = [row for k in sorted(groups) for row in groups[k][:n // len(groups)]]
        else:
            rows = rows[:n]
    with open(out / f'{name}.jsonl', 'w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    print(name, len(rows))
