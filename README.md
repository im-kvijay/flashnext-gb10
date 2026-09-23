# Flash Next on GB10

Work in progress: this is not yet a qualified release.

Target: Qwen3.8-Flash-Next, eight concurrent agents, **at least 200,000 tokens
per agent**, **at least 400 aggregate output tokens/second**, and measured retention of reasoning,
coding, instruction following, tool use, and long-context retrieval.

The baseline is NVIDIA's complete NVFP4 checkpoint, pinned in `runtime.lock.json`.
The NVMe PLE backend retains the checkpoint table bytes and scale, preserves
upstream n-gram hashing, and moves only requested rows into pinned staging memory.
It does not prune experts, change the chat template, truncate requests, or reduce
the requested context. Quantization relative to original BF16 still needs separate
quality evidence. Megakernel-style fusion will be selected using actual profiles.

Code is independent of the earlier Qwen27B training project. No credentials,
checkpoints, rented-machine addresses, or local absolute paths belong in git.

Qualification requires real simultaneous 200k contexts, continuing multi-turn tool
use, nonempty answers, distinct request state, and repeated throughput/latency runs.
A server's context setting or eight HTTP connections alone is insufficient.

## Initial environment

Linux aarch64, GB10, CUDA 13, Python 3.12. Install the pinned vLLM wheel from
`runtime.lock.json`, then `pip install -e .` in the same environment. Set
`FLASHNEXT_PLE_NVME_DIR` to local NVMe storage to enable the plugin.
Without that variable the upstream engine remains unchanged.

The table is an unlinked memory-mapped file populated by the normal model loader;
clean pages are reclaimable and process exit frees its space. Allow roughly 52 GB
extra disk beyond the downloaded checkpoint and runtime. Only gathered rows are
pinned. The initial implementation reloads the table on each server start.

Experimental `FLASHNEXT_PLE_DIRECT=1` instead reads the existing checkpoint
mappings, avoiding that extra table file. It requires a C compiler with OpenMP
and retains exact FP8 bytes. Shard and graph checks and the eight-agent short
full-model screen pass; long-context qualification is pending. Checkpoint export
from this mode is unsupported: retain the original
verified checkpoint.

Optional `FLASHNEXT_ASYNC_PLE=1` copies n-gram IDs into a preallocated pinned
buffer and waits in the gather worker, allowing the decoder thread to enqueue
the first layer. It passed 32 changing direct-PLE graph replays. Its full-model
performance benefit is not yet measured; it remains off by default.

From the cloned repository on a Linux GB10 with CUDA 13 and Python 3.12:

```bash
# Optionally export FLASHNEXT_DATA=/your/nvme/flashnext before both commands.
bash scripts/bootstrap.sh
.venv/bin/python scripts/supervise.py --receipt results/memory-run-001.jsonl
```

The bootstrap installs the locked runtime and verifies the pinned checkpoint's
LFS hashes. It downloads one checkpoint copy. `FLASHNEXT_RUNTIME` selects another
virtual environment path; use that environment's Python for the benchmark tools.
The server listens on localhost:8000 and uses the OpenAI-compatible API. This
development setup is reproducible, but has not met the release requirements.

## Recommended profile: eight agents at 200k

`profiles/gb10-8x200k.env` is the fastest configuration that has passed the
fidelity screen and the eight-way 200k capacity run:

```bash
bash scripts/bootstrap.sh
set -a; . profiles/gb10-8x200k.env; set +a
.venv/bin/python scripts/supervise.py --receipt results/serve-$(date +%s).jsonl
```

Measured at eight concurrent streams on one GB10: 101.6 aggregate output
tokens/s with eight distinct 200k-token contexts, 104.5-109.1 on a natural
coding workload at 4k, 171.6 on short retrieval. Dense weights, experts and
the chat template are unchanged; the GDN recurrent state is stored in BF16
instead of FP32, and the KV cache is FP8. Weight loading takes about 15
minutes. A 200k prompt prefills in about 3 minutes; later turns of the same
conversation reuse the prefix cache. Details: `docs/optimizations.md`.

For development runs use the memory supervisor, with a new receipt filename:

```bash
python scripts/supervise.py --receipt results/memory-run-001.jsonl
```

The default GPU memory fraction is conservatively 0.80 after a higher-budget
long-context run lost contact with its rented host. It is not an eight-agent
capacity claim. Read `docs/qualification.md` for observed outcomes and remaining
release gates, `docs/drafters.md` for DFlash/DFlash 2, and `docs/exl3.md` for the
separate EXL3 candidate. `FLASHNEXT_TEXT_ONLY=1` omits the vision encoder; it is
an explicit text-only option. `FLASHNEXT_KV_DTYPE=fp8` adds KV quantization and
needs paired retention testing. Neither is silently enabled by the launcher.
