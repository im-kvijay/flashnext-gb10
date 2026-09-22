"""Release this checkpoint's clean file cache before GB10 CUDA memory profiling."""
import argparse
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory')
p.add_argument('--also-directory',action='append',default=[],help='Additional owned runtime/install cache directory')
a=p.parse_args()
root=Path(a.directory)
if not (root/'verified-lfs.json').is_file():
    raise SystemExit('Checkpoint verification receipt missing; run scripts/download_model.py first.')
released=0
paths=list(root.glob('*.safetensors'))
for directory in a.also_directory:
    paths.extend(path for path in Path(directory).rglob('*')
                 if path.is_file() and not path.is_symlink() and path.stat().st_size>=1024**2)
for path in paths:
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    released+=path.stat().st_size
print(f'Requested release of clean checkpoint/runtime cache for {released/2**30:.2f} GiB; file contents unchanged.',flush=True)
