"""Bounded warm decode profile; profiler timings are never throughput evidence."""
import argparse
import asyncio
from pathlib import Path
import sys

import aiohttp


async def main(a):
    root = Path(__file__).resolve().parents[1]
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(root / 'bench/concurrency.py'), '--model', a.model,
        '--url', a.url, '--concurrency', '8', '--input-tokens', '4096',
        '--output-tokens', '2048', '--mode', 'performance',
        '--warm-prefixes', '--output', a.output,
    )
    started = False
    try:
        # Eight short primes plus initial scheduling, then a short capture of
        # the long fixed-budget diagnostic. Inspect trace for steady decode.
        await asyncio.sleep(a.delay)
        if process.returncode is not None:
            raise RuntimeError('benchmark exited before profiling could start')
        async with aiohttp.ClientSession() as session:
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--delay',type=float,default=25)
    p.add_argument('--duration',type=float,default=2)
    a = p.parse_args()
    if a.delay <= 0 or not 0 < a.duration <= 30:
        p.error('delay must be positive and capture duration in (0,30] seconds')
    raise SystemExit(asyncio.run(main(a)))
