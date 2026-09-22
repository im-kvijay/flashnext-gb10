"""Start one supervised server configuration and run the standard short screens.

Usage: run_screen.py <label> KEY=VALUE ...   (FLASHNEXT_* overrides on top of the MTP-2 baseline)
Run with the runtime's Python. SCREEN_QUALITY/SCREEN_FIDELITY/SCREEN_ROUTING toggle
screens; SERVER_ARGS adds vLLM arguments; SCREEN_REFERENCE_DIR holds baseline records.
"""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

root = Path(os.environ.get('FLASHNEXT_ROOT', Path(__file__).resolve().parents[1]))
data = Path(os.environ.get('FLASHNEXT_DATA', root / 'data'))
python = os.environ.get('FLASHNEXT_PYTHON', sys.executable)
runtime = os.environ.get('FLASHNEXT_RUNTIME', str(Path(python).parents[1]))
reference = Path(os.environ.get('SCREEN_REFERENCE_DIR', root / 'results/cheaper-direct-mtp2'))
corpus = os.environ.get('SCREEN_CORPUS_ROOT') or subprocess.run(
    [python, '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
    capture_output=True, text=True, check=True).stdout.strip()
label, overrides = sys.argv[1], dict(arg.split('=', 1) for arg in sys.argv[2:])
out = root / 'results' / label
out.mkdir(parents=True, exist_ok=False)
model = data / 'models/nvidia-flashnext'
env = os.environ.copy()
# Same configuration as results/cheaper-direct-mtp2 unless overridden.
env.update(FLASHNEXT_RUNTIME=runtime, FLASHNEXT_DATA=str(data), FLASHNEXT_MODEL=str(model),
           FLASHNEXT_PLE_DIRECT='1', FLASHNEXT_MEMORY_FRACTION='0.60', FLASHNEXT_KV_BYTES=str(6 * 1024**3),
           FLASHNEXT_KV_DTYPE='fp8', FLASHNEXT_TEXT_ONLY='1', FLASHNEXT_MTP='2',
           FLASHNEXT_DRAFT_VOCAB=os.environ.get('FLASHNEXT_DRAFT_VOCAB', str(data / 'draft-vocab-public-vllm.json')), FLASHNEXT_PACK_PLE_STATE='1',
           FLASHNEXT_DIAGNOSTICS_DIR=str(out))
env.update(overrides)
(out / 'configuration.json').write_text(json.dumps(
    {k: v for k, v in env.items() if k.startswith('FLASHNEXT_')}, indent=2) + '\n')
slog = (out / 'server.log').open('x')
server_args = os.environ.get('SERVER_ARGS', '').split()
server = subprocess.Popen([python, 'scripts/supervise.py', '--receipt', str(out / 'memory.jsonl'), *(['--', *server_args] if server_args else [])], cwd=root,
                          env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'supervisor_pid': server.pid}), flush=True)
deadline = time.monotonic() + 3600
started = time.monotonic()
while True:
    if server.poll() is not None:
        raise SystemExit(f'Server exited during startup: {server.returncode}')
    if time.monotonic() > deadline:
        raise SystemExit('Readiness wait expired')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as r:
            if r.status == 200:
                break
    except OSError:
        pass
    time.sleep(5)
print(json.dumps({'ready_seconds': time.monotonic() - started}), flush=True)
m = str(model)
commands = [
    ('retrieval-c8-4k', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '2048', '--mode', 'retrieval']),
    ('throughput-c8-4k', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '256', '--mode', 'performance']),
    ('workload-c8-4k', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '2048', '--mode', 'workload']),
    ('workload-c8-4k-repeat', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '2048', '--mode', 'workload']),
]
if os.environ.get('SCREEN_QUALITY', '1') == '1':
    commands += [
        ('retention', ['bench/retention.py', '--model', m, '--label', label,
                       '--reference', str(reference / 'retention.json')]),
        ('tools-c8-4k', ['bench/tool_continuation.py', '--model', m, '--input-tokens', '4096']),
    ]
if os.environ.get('SCREEN_FIDELITY', '1') == '1':
    commands.append(('fidelity', ['bench/fidelity.py', '--model', m, '--source',
                                  str(reference / 'profile-workload-c8-4k.json'),
                                  '--corpus-root', corpus]))
if os.environ.get('SCREEN_SUITE') == '1':
    commands.append(('suite', ['bench/suite.py', '--data', os.environ.get('FLASHNEXT_EVAL_DATA', str(root / 'data/eval'))]))
if os.environ.get('SCREEN_ROUTING') == '1':
    commands.append(('routing-c8', ['bench/routing_probe.py', '--model', m]))
(out / 'server-args.json').write_text(json.dumps(server_args) + '\n')
results = []
for name, command in commands:
    print('RUN', name, flush=True)
    with (out / (name + '.log')).open('x') as log:
        bench = subprocess.Popen([python, *command, '--output', str(out / (name + '.json'))], cwd=root,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while bench.poll() is None and server.poll() is None:
            time.sleep(2)
        if bench.poll() is None:
            os.killpg(bench.pid, signal.SIGTERM)
            bench.wait()
    results.append({'name': name, 'returncode': bench.returncode})
    (out / 'screen-results.json').write_text(json.dumps(results, indent=2) + '\n')
    if server.poll() is not None or bench.returncode:
        break
    if name == 'retrieval-c8-4k' and json.loads((out / (name + '.json')).read_text())['summary']['retrieval_correct'] != 8:
        break
os.kill(server.pid, signal.SIGTERM)
server.wait()
print(json.dumps(results), flush=True)
print('SCREEN_DONE', flush=True)
