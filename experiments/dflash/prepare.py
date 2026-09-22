"""Create a private source overlay and draft config; never alter the baseline.

Run with the pinned runtime's Python. The overlay symlinks unmodified files,
copies only patched sources, and verifies both input and output hashes.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, draft):
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "manifest.json").read_text())
    if sha(here / "vllm.patch") != manifest["patch_sha256"]:
        raise ValueError("Adapter patch differs from manifest")
    spec = importlib.util.find_spec("vllm")
    original = Path(spec.origin).resolve().parent
    for name, hashes in manifest["files"].items():
        if sha(original.parent / name) != hashes["before"]:
            raise ValueError(f"Unvalidated runtime source: {name}")
    receipt = json.loads((draft / "verified-lfs.json").read_text())
    if receipt["revision"] != manifest["draft_revision"]:
        raise ValueError("Wrong DFlash checkpoint revision")
    output.mkdir(parents=True, exist_ok=False)
    overlay = output / "overlay"
    overlay.mkdir()
    # A file symlink is shared read-only; patched files are replaced with copies.
    def link_file(source, target):
        Path(target).symlink_to(Path(source).resolve())
    shutil.copytree(original, overlay / "vllm", copy_function=link_file,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in manifest["files"]:
        destination = overlay / name
        destination.unlink()
        shutil.copyfile(original.parent / name, destination)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-d", str(overlay),
                    "-i", str(here / "vllm.patch")], check=True)
    for name, hashes in manifest["files"].items():
        if sha(overlay / name) != hashes["after"]:
            raise ValueError(f"Patched source differs: {name}")
    adapted = output / "draft"
    adapted.mkdir()
    for item in draft.iterdir():
        if item.name not in {"config.json", ".cache"}:
            (adapted / item.name).symlink_to(item.resolve())
    shutil.copyfile(here / "draft-config.json", adapted / "config.json")
    shutil.copyfile(here / "dflash_epoch7.py", overlay / "dflash_epoch7.py")
    (output / "adapter-receipt.json").write_text(json.dumps({
        **manifest, "baseline_package": str(original),
        "draft_original": str(draft), "status": "prepared_not_qualified",
    }, indent=2) + "\n")
    print(json.dumps({"PYTHONPATH": str(overlay),
                      "FLASHNEXT_DFLASH_MODEL": str(adapted)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.draft.resolve())
