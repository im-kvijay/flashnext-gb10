"""Start one supervised server configuration and run the standard short screens.

Usage: run_screen.py <label> KEY=VALUE ...   (FLASHNEXT_* overrides on top of the MTP-2 baseline)
Run with the runtime's Python. SCREEN_SPEED/SCREEN_QUALITY/SCREEN_FIDELITY/SCREEN_SUITE/
SCREEN_ROUTING toggle screens; SCREEN_LONG=1 runs only the 8 x 200k capacity check;
SCREEN_PREHOOK names a one-shot GPU script run before the server starts; SERVER_ARGS adds
vLLM arguments; SCREEN_REFERENCE_DIR holds baseline records.
"""
import json, os, re, signal, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
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
prehook = os.environ.get('SCREEN_PREHOOK')
if prehook:
    # Kernel tests and microbenchmarks need the GPU memory a live server holds.
    with (out / 'prehook.log').open('x') as log:
        subprocess.run(['bash', prehook], stdout=log, stderr=subprocess.STDOUT, timeout=3600)
env = os.environ.copy()
# Same configuration as results/cheaper-direct-mtp2 unless overridden.
env.update(FLASHNEXT_RUNTIME=runtime, FLASHNEXT_DATA=str(data), FLASHNEXT_MODEL=str(model),
           FLASHNEXT_PLE_DIRECT='1', FLASHNEXT_MEMORY_FRACTION='0.60', FLASHNEXT_KV_BYTES=str(6 * 1024**3),
           FLASHNEXT_KV_DTYPE='fp8', FLASHNEXT_TEXT_ONLY='1', FLASHNEXT_MTP='2',
           FLASHNEXT_DRAFT_VOCAB=os.environ.get('FLASHNEXT_DRAFT_VOCAB', str(data / 'draft-vocab-public-vllm.json')), FLASHNEXT_PACK_PLE_STATE='1',
           FLASHNEXT_DIAGNOSTICS_DIR=str(out))
env.update(overrides)
# More concurrent eval sequences need proportionally more KV blocks (each long
# reasoning request holds about 18 blocks of 45 MB with MTP-3 state slots).
if 'FLASHNEXT_KV_BYTES' not in overrides and int(env.get('FLASHNEXT_SEQUENCES', '8')) > 8:
    env['FLASHNEXT_KV_BYTES'] = str(int(int(env['FLASHNEXT_SEQUENCES']) * 0.875 * 1024**3))
# An editable install elsewhere must not shadow this tree.
env['PYTHONPATH'] = str(root) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
(out / 'configuration.json').write_text(json.dumps(
    {k: v for k, v in env.items() if k.startswith('FLASHNEXT_')}, indent=2) + '\n')
slog = (out / 'server.log').open('x')
server_args = os.environ.get('SERVER_ARGS', '').split()
server = subprocess.Popen([python, 'scripts/supervise.py', '--receipt', str(out / 'memory.jsonl'), *(['--', *server_args] if server_args else [])], cwd=root,
                          env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'supervisor_pid': server.pid}), flush=True)


def prefetch_shards(model_dir, log_path, stop):
    """Read the loader's next checkpoint shards in parallel ahead of it.

    The loader faults shard pages in one at a time (about 0.1-0.3 GB/s on an
    overlay filesystem); 12 parallel 64 MiB preads reach over 1 GB/s. Follows the
    loader's progress bar, stays at most two shards ahead, and drops finished
    shards from the page cache. The PLE shard is read on demand, so it is skipped.
    """
    files = sorted(Path(model_dir).glob('*.safetensors'))
    progress = re.compile(r'Loading safetensors checkpoint shards:\s+\d+% Completed \| (\d+)/(\d+)')
    read = set()
    pool = ThreadPoolExecutor(12)

    def read_file(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            list(pool.map(lambda off: len(os.pread(fd, 64 << 20, off)), range(0, os.fstat(fd).st_size, 64 << 20)))
        finally:
            os.close(fd)

    def drop(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    loaded = 0
    while not stop.is_set():
        try:
            found = progress.findall(Path(log_path).read_text(errors='replace')[-20000:])
        except OSError:
            found = []
        if found:
            loaded, total = map(int, found[-1])
            if loaded >= total:
                break
        for path in files[:max(loaded - 1, 0)]:
            if path in read:
                drop(path)
                read.discard(path)
        pending = [f for f in files[loaded:loaded + 2] if 'ple' not in f.name and f not in read]
        if pending:
            read_file(pending[0])
            read.add(pending[0])
        else:
            stop.wait(1)
    pool.shutdown(wait=False)


prefetch_stop = threading.Event()
if os.environ.get('SCREEN_PREFETCH', '1') == '1':
    threading.Thread(target=prefetch_shards, args=(env['FLASHNEXT_MODEL'], out / 'server.log', prefetch_stop),
                     daemon=True).start()
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
prefetch_stop.set()
print(json.dumps({'ready_seconds': time.monotonic() - started}), flush=True)
m = str(model)
commands = [
    ('retrieval-c8-4k', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '2048', '--mode', 'retrieval']),
    ('throughput-c8-4k', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '256', '--mode', 'performance']),
    ('workload-c8-4k', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '2048', '--mode', 'workload']),
    ('workload-c8-4k-repeat', ['bench/concurrency.py', '--model', m, '--input-tokens', '4096', '--output-tokens', '2048', '--mode', 'workload']),
]
long_only = os.environ.get('SCREEN_LONG') == '1'
if os.environ.get('SCREEN_SPEED') == '0':
    commands = []
if long_only:
    # Capacity qualification only: eight distinct 200k-token codebase contexts.
    commands = [('codebase-c8-200k', ['bench/concurrency.py', '--model', m, '--input-tokens', '200000',
                                      '--output-tokens', '2048', '--mode', 'codebase', '--corpus-root', corpus])]
if os.environ.get('SCREEN_QUALITY', '1') == '1' and not long_only:
    commands += [
        ('retention', ['bench/retention.py', '--model', m, '--label', label,
                       '--reference', str(reference / 'retention.json')]),
        ('tools-c8-4k', ['bench/tool_continuation.py', '--model', m, '--input-tokens', '4096']),
    ]
if os.environ.get('SCREEN_FIDELITY', '1') == '1' and not long_only:
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
