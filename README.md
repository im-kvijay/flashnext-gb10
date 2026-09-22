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

Deployment commands and measured results will be finalized after hardware tests.
