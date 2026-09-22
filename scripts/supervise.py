"""Launch this service with a host-memory floor and durable memory readings."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def available_bytes():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main(a):
    root = Path(__file__).resolve().parents[1]
    threshold = int(a.minimum_available_gib * 1024**3)
    if available_bytes() < threshold:
        raise SystemExit('Insufficient host memory before launch; stop competing workloads or lower the GPU budget.')
    output = Path(a.receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the receipt before launching an expensive service.
    stream = output.open('x')
    try:
        process = subprocess.Popen(['bash',str(root/'scripts/serve.sh'),*a.server_args],start_new_session=True)
    except BaseException:
        stream.close()
        raise
    low_since = None
    tripped = False
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with stream:
            while process.poll() is None and not stopping:
                available = available_bytes()
                now = time.monotonic()
                low_since = (low_since or now) if available < threshold else None
                row = {'time_unix':time.time(),'service_pid':process.pid,
                       'mem_available_bytes':available,'minimum_bytes':threshold}
                if low_since is not None and now-low_since >= a.grace_seconds:
                    row['failure'] = 'host_memory_floor'
                    tripped = True
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                if tripped:
                    print('Host memory floor reached: stopping this service. Lower its memory budget before retrying.',flush=True)
                    break
                time.sleep(1)
    finally:
        terminate_group(process)
    return 75 if tripped else (process.returncode or 0)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',required=True,help='New JSONL file; existing runs are never overwritten')
    p.add_argument('--minimum-available-gib',type=float,default=8)
    p.add_argument('--grace-seconds',type=float,default=3)
    a, a_server=p.parse_known_args()
    a.server_args=a_server[1:] if a_server[:1]==['--'] else a_server
    if a.minimum_available_gib<=0 or a.grace_seconds<0:
        p.error('memory floor must be positive and grace nonnegative')
    raise SystemExit(main(a))
