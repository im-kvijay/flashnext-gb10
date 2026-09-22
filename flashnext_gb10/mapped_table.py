"""Reclaimable, exact-byte table storage. Only lookup results are pinned."""
from __future__ import annotations

import mmap
import os
from pathlib import Path
import shutil
import tempfile


class MappedTable:
    def __init__(self, directory: str, size: int):
        if size <= 0:
            raise ValueError("table size must be positive")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(path).free < size + 2 * 1024**3:
            raise RuntimeError(f"NVMe table needs {size / 1024**3:.2f} GiB plus 2 GiB free in {path}")
        fd, filename = tempfile.mkstemp(prefix="ple-", suffix=".bin", dir=path)
        self.size = size
        self.path = filename
        try:
            os.ftruncate(fd, size)
            self.mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            # The mapping remains valid; process death automatically frees its disk.
            os.unlink(filename)
        finally:
            os.close(fd)
        self.mapping.madvise(mmap.MADV_RANDOM)

    def flush_rows(self, offset: int, length: int):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("flush range outside mapped table")
        if not length:
            return
        start = offset // mmap.PAGESIZE * mmap.PAGESIZE
        end = min(self.size, (offset + length + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
        self.mapping.flush(start, end - start)
        self.mapping.madvise(mmap.MADV_DONTNEED, start, end - start)
