# Agent runbook: set up Flash Next 8 x 200k on a GB10

You are setting up an OpenAI-compatible server for NVIDIA's
Qwen3.8-Flash-Next-NVFP4 on one GB10 (DGX Spark class) from a plain `git clone`
of this repository. Everything needed ships in the clone: the retrained MTP
drafter and the reference records are under `assets/` and are unpacked and
checksum-verified by the setup script. No release download is needed. Run the
steps in order, check each against its success criterion, and report the
numbers listed at the end.

Expected result: eight concurrent agents with 212,992 tokens of context each,
about 120-128 output tokens/s aggregate at 4k and about 131 at 200k, outputs
matching the reference host's numerics.

## 0. Prerequisites (check; ask the human before changing the system)

```bash
uname -m                                   # aarch64
nvidia-smi --query-gpu=name --format=csv   # NVIDIA GB10
python3.12 --version                       # 3.12.x
free -g                                    # total about 121 GiB
df -h .                                    # about 160 GB free where the data will live
```

Missing packages: `sudo apt install python3.12 python3.12-venv python3.12-dev build-essential curl`
(needs the human's approval for sudo). Network access to Hugging Face,
download.pytorch.org, flashinfer.ai and PyPI is required. If Hugging Face
refuses the download, ask the human for a token and `export HF_TOKEN=...`
(never write it into a file in the repository).

Other GPU jobs and other vLLM servers must be stopped for the whole session:
this server uses about 111 GiB of the 121 GiB unified memory.
`nvidia-smi` should list no other compute processes.

## 1. Setup (30-90 minutes, mostly the 124 GB checkpoint download)

Pick a directory on local NVMe with about 160 GB free for the checkpoint and
caches (the default `./data` inside the clone is fine if the clone is on NVMe).
If the machine already has the checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4`,
add `--model DIR` (its hashes are verified; the download is skipped).

```bash
bash scripts/setup_gb10.sh --data /path/on/nvme/flashnext 2>&1 | tee setup.log
```

It is safe to re-run after a failure; finished steps are skipped and downloads
resume. It never modifies an existing vLLM install. It creates `./.venv` with the
exact pinned vLLM build, because the plugin only works on that build.

Success, all visible in `setup.log`:
- `Release bundle files verified`
- `Drafter ... matches the released weights (trained2-576gen-2026-09-24)`
- `==> Done`, followed by the contents of `flashnext.local.env`

## 2. Start the server (first start 20-30 minutes, later about 15)

```bash
bash scripts/start.sh --background
bash scripts/wait_ready.sh          # blocks until /health is 200; exits 1 with the log tail if the server died
```

`start.sh` prints the available memory and, if the machine has less than the
full profile needs (about 118 GiB available at start), shrinks the budget and
prints a `Memory fit:` line: smaller PLE cache, then smaller prefill chunks,
then fewer agents. Outputs are unaffected. Record that line if it appears.

- Endpoint: `http://127.0.0.1:8000/v1`
- Model name: `flashnext`
- Server log: `results/serve-latest.log`
- Stop: `bash scripts/stop.sh`

Eight is the maximum number of requests decoded together, not a requirement.
Fewer active agents just run in smaller batches, each generating faster; a
ninth request waits in the queue. Leave the server running for steps 3-5.

## 3. Smoke test (about 5 minutes; `--long` adds about 30)

```bash
bash scripts/smoke_test.sh
bash scripts/smoke_test.sh --long
```

Success:
- The chat answer is `391`.
- A `read_file` tool call is printed.
- Ends with `OK.`
- At 4k: about 120-128 tok/s aggregate, accepted length about 2.5.
- With `--long` (8 x 200k): about 131 tok/s, accepted length about 2.7.

Up to about 15% lower is normal on slower storage or with a desktop session
running. Well below that means something is misconfigured (see Troubleshooting).

## 4. Numerics check (about 15 minutes)

```bash
bash scripts/verify_fidelity.sh
```

Success: the last line is `PASS`. The reference host measured 96.5% top-1 and
KL 0.007 on coding-agent text, and 85.8% on raw source, against the
unquantized-dense reference.

## 5. Quality check against the base model

```bash
bash scripts/quality_check.sh --quick   # about 20 minutes: everyday tasks, 8-agent tool use, long-context retrieval
bash scripts/quality_check.sh           # about 2 hours: adds LiveCodeBench, HumanEval, GSM8K, MMLU-Pro
```

Reference results (base model served with stock vLLM settings / this recipe):

| Check | Base | Recipe | Treat as a failure |
|---|---|---|---|
| everyday tasks | 24/24 | 24/24 | below 22/24, or any category 0/3 |
| tool use, 8 agents | 8/8 | 8/8 | below 8/8 |
| retrieval | 10/12 | 12/12 | below 10/12 |
| HumanEval | 37/40 | 37/40 | below 35 |
| GSM8K | 20/20 | 20/20 | below 19 |
| MMLU-Pro | 22/28 | 21/28 | below 19 |
| LiveCodeBench (16k budget) | 12/30 | 9/30 | not a gate |

LiveCodeBench at 16k tokens is too noisy to gate on: nearly every failure is
an answer cut off at the budget, and runs of the same setup range 7-12
(`docs/validation.md`). `bench/suite.py --compare` prints the per-item
differences.

## Do not

- Do not upgrade, downgrade or reinstall vLLM, torch or flashinfer, and do not
  `pip install` into `.venv` beyond what the scripts do. The plugin patches
  internals of the exact pinned build.
- Do not run a second vLLM server or other GPU jobs while serving; `start.sh`
  refuses if a vLLM server is already running.
- Do not change the numerics settings in `profiles/gb10-8x200k.env`:
  `FLASHNEXT_MOE_BACKEND`, `FLASHNEXT_DENSE_W8A16`, `FLASHNEXT_KV_DTYPE`,
  `FLASHNEXT_SSM_DTYPE`. They are what `verify_fidelity.sh` checks.
- Do not enable `FLASHNEXT_GDN_REPLAY`: it is validated for numerics but still
  awaits its task-suite comparison.
- Do not raise `FLASHNEXT_PREFILL` above 2048. At 4096 host memory dipped below
  the 6 GiB safety floor with eight 200k contexts.
- Do not disable the memory supervisor or lower `FLASHNEXT_MIN_AVAILABLE_GIB`.
  Running out of unified memory hangs the whole machine.
- Do not stop the desktop session (`systemctl isolate multi-user.target`)
  without the human's approval.
- Do not bind to `0.0.0.0` without `--api-key`: `bash scripts/start.sh --background --api-key KEY`
  plus `FLASHNEXT_HOST=0.0.0.0` in `flashnext.local.env`.
- Do not commit `flashnext.local.env`, `release/`, `results/` or tokens.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `setup_gb10.sh` dies in Preflight | install the named package (with the human's approval); memory total below 115 GiB means this is not a 128 GB GB10 |
| download stalls or 401 | re-run setup (it resumes); `export HF_TOKEN=...` |
| `release bundle files do not match` | `rm -rf release` and re-run setup (re-unpacks from `assets/`); if it persists, `git status` to check `assets/` is unmodified |
| `wait_ready.sh`: server exited | read `results/serve-latest.log` from the first `Error`/`Traceback`. Exit code 75 or `host_memory_floor` in `results/serve-*.jsonl` means memory ran out: stop other workloads and restart (the start will fit fewer agents) |
| `Not enough memory for this configuration` | something else holds memory: `nvidia-smi`, `free -g`, stop it |
| `Direct PLE lookup needs a C compiler` | `sudo apt install build-essential` |
| throughput far below the numbers above | check `nvidia-smi` for other processes, that `--data` is on local NVMe (not a network or loop mount), and that setup reported the released drafter |
| `verify_fidelity.sh` FAIL | profile edited or different vLLM build: `git diff profiles/`, and check the setup log's vLLM version |
| first start takes more than 45 minutes | normal only on the first start (kernel compiles, cached under the data directory); otherwise check the log |

## Report back

- `nvidia-smi` GPU name, and memory available at start (the `Memory:` line from
  `start.sh`, plus any `Memory fit:` line).
- The drafter line from `setup.log`.
- Smoke test 4k and 200k tok/s and accepted lengths.
- The `verify_fidelity.sh` result lines.
- Quality-check scores compared with the table above.
- Any deviation from these steps, and why.
