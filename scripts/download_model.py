"""Download one pinned copy, then verify LFS hashes before serving."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--directory", required=True)
a = p.parse_args()
lock = json.loads((Path(__file__).resolve().parents[1] / "runtime.lock.json").read_text())
snapshot_download(lock["model"], revision=lock["model_revision"], local_dir=a.directory, max_workers=8)
info = HfApi().model_info(lock["model"], revision=lock["model_revision"], files_metadata=True)
verified = []
for entry in info.siblings:
    if entry.lfs is None:
        continue
    path = Path(a.directory) / entry.rfilename
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != entry.lfs.sha256:
        raise RuntimeError(f"SHA256 mismatch: {entry.rfilename}")
    verified.append({"path": entry.rfilename, "sha256": digest, "size": path.stat().st_size})
receipt = {"model": lock["model"], "revision": info.sha, "verified": verified}
(Path(a.directory) / "verified-lfs.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(f"Verified {len(verified)} LFS files at {info.sha}", flush=True)
