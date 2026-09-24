# Setting up Flash Next 8 x 200k on a GB10

This guide takes a GB10 (DGX Spark class, 128 GB unified memory, aarch64,
CUDA 13 driver) from nothing to an OpenAI-compatible server for eight
concurrent agents with 212,992 tokens of context each. The server runs
NVIDIA's Qwen3.8-Flash-Next-NVFP4 checkpoint with the tuning measured in
`docs/optimizations.md`. Every step is a script.

## Quick start

From a release bundle (includes the retrained drafter and reference records):

```bash
tar xzf flashnext-gb10-<commit>.tar.gz
cd flashnext-gb10-<commit>
bash scripts/setup_gb10.sh --data /path/on/nvme/flashnext   # 30-90 min, mostly the 124 GB download
bash scripts/start.sh                                         # first start 20-30 min (kernel compiles), then ~15
# in another shell:
bash scripts/smoke_test.sh                                    # chat, tool call, 8 agents at 4k
bash scripts/verify_fidelity.sh                               # numerics match the reference host
bash scripts/smoke_test.sh --long                             # 8 distinct 200k contexts
```

From a plain clone (no drafter file): the same commands work. The server then
uses the checkpoint's own MTP drafter (identical outputs, lower throughput).
Add `--build-drafter` to retrain it on this machine (2-4 hours of GPU time,
`scripts/build_drafter.sh`). `verify_fidelity.sh` needs the bundle's
reference records.

The endpoint is `http://127.0.0.1:8000/v1`, model name `flashnext`. Tool
calling uses the official Qwen XML parser (`--tool-call-parser qwen3_xml`)
with auto tool choice, and the reasoning parser separates `<think>` content
into `reasoning_content`. Any OpenAI-compatible agent harness works.

## Requirements

| Item | Needed |
|---|---|
| Machine | one GB10, 128 GB unified memory, Linux aarch64, NVIDIA driver with CUDA 13 |
| Packages | `python3.12 python3.12-venv python3.12-dev build-essential curl` |
| Disk | about 160 GB free on local NVMe: 124 GB checkpoint, 16 GB runtime, caches; drafter rebuild adds about 60 GB temporarily |
| Network | Hugging Face (checkpoint), the vLLM, PyTorch and FlashInfer wheel indexes |
| Memory at idle | at least 112 GiB available; stop the desktop session and other GPU jobs |

**Existing vLLM installs.** The plugin patches vLLM internals, so it only runs
on the exact vLLM commit in `runtime.lock.json`
(`0.29.1rc1.dev518+ga33b3bac5`). `setup_gb10.sh` never modifies an existing
vLLM: it reports what it found and creates `./.venv` with the pinned wheels.
If an existing environment already has that exact build, pass
`--reuse-vllm /path/to/env/bin/python` to install only the plugin into it.

**Existing checkpoint.** `--model DIR` reuses a download of
`nvidia/Qwen3.8-Flash-Next-NVFP4` at revision `fc694b54...` (hashes are
verified against Hugging Face, then recorded in `DIR/verified-lfs.json`).

## What the scripts do

| Script | Does |
|---|---|
| `scripts/setup_gb10.sh` | preflight (GPU, memory, disk, compiler), pinned runtime, plugin install and import check, checkpoint download plus SHA-256 verification, drafter install (bundle file, `--drafter FILE`, or `--build-drafter`), writes `flashnext.local.env`. Safe to re-run. |
| `scripts/start.sh` | reads `profiles/gb10-8x200k.env`, then `flashnext.local.env`, then your exported variables; starts vLLM under `scripts/supervise.py`, which stops the server if host memory stays below the floor (6 GiB) instead of letting the machine lock up. Extra arguments go to `vllm serve` (e.g. `--api-key KEY`). |
| `scripts/smoke_test.sh` | waits for `/health`, checks a chat answer and a tool call, runs eight concurrent 4k coding agents and prints throughput and draft acceptance; `--long` adds eight distinct 200k-token codebase contexts. |
| `scripts/verify_fidelity.sh` | teacher-forced scoring of 16 sequences against the reference host's records: top-1 agreement and KL against the unquantized-dense reference and against the same profile on the reference host. |
| `scripts/build_drafter.sh` | KodCode prompts, target generations, drafter-input capture, MTP weight extraction and training (`experiments/drafter/`). |
| `scripts/make_release.sh` | builds a bundle: `git archive` of a commit plus `release/` (drafter, reference records, SHA256SUMS). |

## Expected results (reference host)

Measured on a rented GB10 whose model storage sits behind a loop device
(slower per-layer-embedding reads than a local NVMe filesystem):

| Measurement | Result |
|---|---|
| 8 agents, 4k context, natural coding workload | about 120-128 aggregate output tok/s, accepted length about 2.5 of 4 |
| 8 agents, 200k distinct codebase contexts, 2k continuations | 132.7 tok/s with the 1x drafter (see `docs/optimizations.md`); final profile: see the table there |
| Cold 200k-token prefill (one agent) | 90 s, 2,212 tok/s |
| Eight concurrent 12k-token appends at 200k depth | 3,221 tok/s |
| KV capacity | 1,652,167 tokens (7.76 x 212,992); 8 x 200k fit |
| Fidelity vs unquantized dense (BF16 KV) | coding-agent continuations: top-1 96.6%, KL 0.007; raw source code: 86.7%, KL 0.20 |

No activations are quantized. Routed experts keep NVIDIA's NVFP4 weights with
BF16 activations (Marlin W4A16); the GDN/QSA projections use FP8 weights with
BF16 activations; lm_head, embeddings, shared experts and the chat template are
unchanged; KV cache FP8, GDN state BF16. The raw-source-code agreement is
limited by the model's own run-to-run sensitivity on that input (served vs
served on identical requests differs as much as served vs an FP32 reference,
`docs/optimizations.md`).

## Operating notes

- **One server at a time.** Two vLLM processes do not fit; `start.sh` refuses
  to start a second one. Do not run other GPU jobs while serving.
- **Memory floor.** If host available memory stays below
  `FLASHNEXT_MIN_AVAILABLE_GIB` (6), the supervisor stops the server
  (exit 75) rather than risk an out-of-memory hang. With a desktop session or
  other services using several GB, run fewer agents: set
  `FLASHNEXT_SEQUENCES=6` and `FLASHNEXT_KV_BYTES=19730006016` in
  `flashnext.local.env` (six agents at 212,992 tokens).
- **Prefix caching** is on: later turns of the same conversation reuse the
  cached prefix, so only new tokens are prefilled.
- **Network access.** The server binds to 127.0.0.1. To serve other machines
  set `FLASHNEXT_HOST=0.0.0.0` in `flashnext.local.env` and pass
  `--api-key` to `start.sh`; there is no other authentication.
- **Logs.** `results/serve-<time>.log` (server) and `.jsonl` (memory
  receipts, one per second).
- **Stopping.** Ctrl-C in the `start.sh` shell, or `pkill -f supervise.py`.
- **Restarts** take about 15 minutes (weight loading) plus graph capture;
  compiled kernels are cached under `$FLASHNEXT_DATA/cache`.

## Configuration reference

All settings live in `profiles/gb10-8x200k.env`, commented. The ones that
change numerics (and passed the fidelity gate) are `FLASHNEXT_MOE_BACKEND=marlin`,
`FLASHNEXT_DENSE_W8A16=1`, `FLASHNEXT_KV_DTYPE=fp8` and
`FLASHNEXT_SSM_DTYPE=bfloat16`. Everything else is lossless scheduling, I/O or
kernel choice. `FLASHNEXT_MTP_OVERRIDE` (set by setup) points at the retrained
drafter; the drafter only proposes tokens, so it cannot change outputs.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `setup_gb10.sh`: vLLM version mismatch | expected: a separate `.venv` is created; `--reuse-vllm` only accepts the exact pinned build |
| download stops | re-run `setup_gb10.sh`; downloads resume; `HF_TOKEN` if Hugging Face asks for login |
| server exits with 75 | memory floor tripped: stop other workloads or run six agents (above) |
| `Direct PLE lookup needs a C compiler` | `sudo apt install build-essential` |
| first start very slow | Triton/FlashInfer compile on first start; cached afterwards |
| `smoke_test.sh` tool call missing | check the server log for parser errors; the chat template must be the checkpoint's own |
| `verify_fidelity.sh` FAIL | different vLLM build or profile edits; compare `results/*/configuration.json` with the bundle's |
