"""Bounded warm decode profile; profiler timings are never throughput evidence."""
import argparse
import asyncio
from pathlib import Path
import sys
import tempfile
import time

import aiohttp


async def main(a):
    root = Path(__file__).resolve().parents[1]
    markers = tempfile.TemporaryDirectory(prefix='flashnext-profile-')
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(root / 'bench/concurrency.py'), '--model', a.model,
        '--url', a.url, '--concurrency', '8', '--input-tokens', '4096',
        '--output-tokens', '4096', '--mode', 'workload',
        '--warm-prefixes', '--first-token-dir', markers.name, '--output', a.output,
    )
    started = False
    try:
        deadline = time.monotonic()+a.ready_timeout
        async with aiohttp.ClientSession() as session:
            while True:
                if process.returncode is not None:
                    raise RuntimeError('benchmark exited before all eight streams were ready')
                if time.monotonic()>deadline:
                    raise RuntimeError('all-eight decode readiness timed out')
                if len(list(Path(markers.name).glob('agent-*.json')))==8:
                    async with session.get(a.url+'/metrics') as response:
                        response.raise_for_status()
                        body=await response.text()
                    running=sum(float(line.rsplit(' ',1)[1]) for line in body.splitlines()
                                if line.startswith('vllm:num_requests_running{'))
                    if running==8:
                        break
                await asyncio.sleep(0.5)
            print('All eight streams generating; starting bounded profile',flush=True)
            async with session.post(a.url + '/start_profile') as response:
                response.raise_for_status()
                started = True
            try:
                await asyncio.sleep(a.duration)
            finally:
                if started:
                    async with session.post(a.url + '/stop_profile') as response:
                        response.raise_for_status()
        return await process.wait()
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        markers.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--ready-timeout',type=float,default=600)
    p.add_argument('--duration',type=float,default=2)
    a = p.parse_args()
    if a.ready_timeout <= 0 or not 0 < a.duration <= 30:
        p.error('readiness timeout must be positive and capture duration in (0,30] seconds')
    raise SystemExit(asyncio.run(main(a)))
