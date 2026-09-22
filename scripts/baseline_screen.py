"""Run the first fidelity and speed screens once this server is ready."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output-dir', required=True)
p.add_argument('--ready-timeout', type=int, default=1200)
a = p.parse_args()
deadline = time.monotonic() + a.ready_timeout
while True:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as r:
            if r.status == 200:
                break
    except OSError:
        pass
    if time.monotonic() > deadline:
        raise SystemExit('Server readiness timed out; no benchmarks run')
    time.sleep(5)
root = Path(__file__).resolve().parents[1]
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
for name, concurrency, tokens, output, mode in [
    ('retrieval-c8-4k', 8, 4096, 2048, 'retrieval'),
    ('throughput-c8-4k', 8, 4096, 256, 'performance'),
    ('retrieval-c1-200k', 1, 200000, 2048, 'retrieval'),
]:
    cmd = [sys.executable, str(root/'bench/concurrency.py'), '--model', a.model,
           '--concurrency', str(concurrency), '--input-tokens', str(tokens),
           '--output-tokens', str(output), '--mode', mode, '--output', str(out/(name+'.json'))]
    print('RUN', name, flush=True)
    subprocess.run(cmd, check=True)
    report = json.loads((out/(name+'.json')).read_text())
    if mode == 'retrieval' and report['summary']['retrieval_correct'] != concurrency:
        raise SystemExit(f'{name}: semantic screen failed; inspect outputs before proceeding')
print('BASELINE_SCREEN_FINISHED', flush=True)
