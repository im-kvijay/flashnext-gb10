"""Byte-exact CPU lookup directly from safetensors mappings, with no table copy."""
import ctypes
import hashlib
import mmap
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import torch


def _mapped_file_region(tensor):
    first = tensor.data_ptr()
    last = first + tensor.numel() * tensor.element_size()
    cursor = first
    identity = None
    file_delta = None
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        lo,hi = (int(x,16) for x in fields[0].split('-'))
        if hi <= cursor:
            continue
        if lo > cursor:
            break
        if len(fields)!=6 or not fields[5].endswith('.safetensors') or fields[4]=='0':
            raise ValueError('PLE source must be entirely inside a live safetensors file mapping')
        if 'r' not in fields[1]:
            raise ValueError('PLE source mapping is not readable')
        # MADV_RANDOM can split a file's VMA at page boundaries. A following
        # tensor may start in the last advised page and continue in the next
        # VMA. Require contiguous bytes of the same file, not a single VMA.
        current_identity = (fields[3],fields[4],fields[5])
        current_delta = int(fields[2],16)-lo
        if identity is None:
            identity,file_delta = current_identity,current_delta
        elif current_identity!=identity or current_delta!=file_delta:
            raise ValueError('PLE source crosses unrelated file mappings')
        cursor=min(hi,last)
        if cursor==last:
            return first,last,identity[2],first+file_delta
    raise ValueError('PLE source is not file-backed; eager checkpoint loading is unsupported')


def _load_lookup(directory):
    source=Path(__file__).with_name('gather_shards.c')
    key=hashlib.sha256(source.read_bytes()+platform.machine().encode()).hexdigest()[:24]
    root=Path(directory)/'lookup-kernel'
    root.mkdir(parents=True,exist_ok=True)
    library=root/f'gather-{key}.so'
    if not library.exists():
        compiler=shutil.which('cc')
        if compiler is None:
            raise RuntimeError('Direct PLE lookup needs a C compiler; install build-essential or use the copied-table backend')
        fd,name=tempfile.mkstemp(suffix='.so',dir=root)
        os.close(fd)
        try:
            subprocess.run([compiler,'-O3','-shared','-fPIC','-fopenmp',str(source),'-o',name],check=True)
            os.replace(name,library)
        finally:
            Path(name).unlink(missing_ok=True)
    dll=ctypes.CDLL(str(library))
    function=dll.flashnext_gather_shards
    function.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64,ctypes.c_void_p,
        ctypes.c_int64,ctypes.c_int64,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
    function.restype=None
    reader=dll.flashnext_pread_shards
    reader.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64,ctypes.c_void_p,
        ctypes.c_int64,ctypes.c_int64,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    reader.restype=ctypes.c_int
    return dll,function,reader


class CheckpointShards:
    def __init__(self,width,dtype,directory,threads=4,io_mode=None,io_threads=None):
        io_mode=io_mode or os.environ.get('FLASHNEXT_PLE_IO','buffered')
        io_threads=io_threads or int(os.environ.get('FLASHNEXT_PLE_IO_THREADS','32'))
        if (dtype != torch.float8_e4m3fn or width<=0 or not 1<=threads<=16 or not 1<=io_threads<=128
                or io_mode not in ('buffered','direct','mapped')):
            raise ValueError('Direct PLE lookup supports FP8 rows, 1-16 copy threads, 1-128 I/O threads and buffered/direct/mapped I/O')
        self.width,self.dtype,self.threads,self.io_threads=width,dtype,threads,io_threads
        # Mapped copies fault only a few pages at a time; one step's hundreds of
        # random cold rows then stall decoding. pread keeps many reads in flight:
        # buffered keeps page-cache hits, direct bypasses the cache. Mapped is
        # the fallback and the byte reference.
        self.io_mode=io_mode
        self.direct_io=io_mode!='mapped'
        self.parts=[]
        self.sealed=False
        self._fds={}
        self._dll,self._lookup,self._pread=_load_lookup(directory)

    def add(self,start,tensor):
        if self.sealed:
            raise RuntimeError('PLE shard reload requires a service restart')
        if (start<0 or tensor.device.type!='cpu' or tensor.dtype!=self.dtype
                or tensor.ndim!=2 or tensor.shape[1]!=self.width or not tensor.is_contiguous()):
            raise ValueError('Invalid direct PLE checkpoint shard')
        if tensor.shape[0]==0:
            return
        first,last,path,file_offset=_mapped_file_region(tensor)
        page=mmap.PAGESIZE
        begin=first//page*page
        end=(last+page-1)//page*page
        libc=ctypes.CDLL(None,use_errno=True)
        libc.madvise.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
        libc.madvise.restype=ctypes.c_int
        if libc.madvise(begin,end-begin,mmap.MADV_RANDOM):
            raise OSError(ctypes.get_errno(),'Unable to set random-access advice for PLE mapping')
        # Keeping the tensor alive retains its underlying safetensors mapping.
        self.parts.append((start,start+tensor.shape[0],tensor,path,file_offset))

    def seal(self,valid_rows):
        ordered=sorted(self.parts,key=lambda x:x[0])
        end=0
        for start,next_end,*_ in ordered:
            if start!=end:
                raise ValueError('PLE checkpoint shards overlap or contain a gap')
            end=next_end
        if end!=valid_rows:
            raise ValueError('PLE checkpoint shards do not cover the real vocabulary')
        self.parts=ordered
        self.valid_rows=valid_rows
        n=len(ordered)
        self._pointers=(ctypes.c_void_p*n)(*(p[2].data_ptr() for p in ordered))
        self._starts=(ctypes.c_int64*n)(*(p[0] for p in ordered))
        self._ends=(ctypes.c_int64*n)(*(p[1] for p in ordered))
        if self.direct_io:
            try:
                for path in {p[3] for p in ordered}:
                    flags=os.O_RDONLY|(os.O_DIRECT if self.io_mode=='direct' else 0)
                    self._fds[path]=os.open(path,flags)
                    if self.io_mode=='buffered':
                        os.posix_fadvise(self._fds[path],0,0,os.POSIX_FADV_RANDOM)
            except OSError:
                # Filesystems such as tmpfs refuse O_DIRECT; use mapped copies.
                for fd in self._fds.values():
                    os.close(fd)
                self._fds={}
                self.direct_io=False
                self.io_mode='mapped'
        if self.direct_io:
            self._fd_array=(ctypes.c_int*n)(*(self._fds[p[3]] for p in ordered))
            self._bases=(ctypes.c_int64*n)(*(p[4] for p in ordered))
        self.sealed=True

    def gather_into(self,ids,output,valid_rows):
        if not self.sealed:
            self.seal(valid_rows)
        if valid_rows!=self.valid_rows:
            raise ValueError('PLE vocabulary changed after loading')
        if ids.device.type!='cpu':
            raise ValueError('Direct PLE lookup requires CPU IDs')
        flat=ids.reshape(-1).to(torch.int64).contiguous()
        if (output.device.type!='cpu' or output.dtype!=torch.uint8 or not output.is_contiguous()
                or output.numel()!=flat.numel()*self.width):
            raise ValueError('Direct PLE output buffer has wrong device, dtype, size or layout')
        if self.direct_io:
            failures=self._pread(self._fd_array,self._bases,self._starts,self._ends,len(self.parts),
                                 flat.data_ptr(),flat.numel(),valid_rows,self.width,output.data_ptr(),
                                 self.io_threads,int(self.io_mode=='direct'))
            if failures:
                raise OSError(f'{failures} direct PLE row reads failed')
            return
        self._lookup(self._pointers,self._starts,self._ends,len(self.parts),flat.data_ptr(),
                     flat.numel(),valid_rows,self.width,output.data_ptr(),self.threads)
