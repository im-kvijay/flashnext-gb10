"""Check direct-file PLE byte patterns, ownership, coverage, and parameter shape."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.model_executor.parameter import ModelWeightParameter
from flashnext_gb10.checkpoint_shards import CheckpointShards

with tempfile.TemporaryDirectory() as directory:
    # Includes every FP8 storage bit pattern, including non-finite values.
    source=torch.arange(256,dtype=torch.uint8).repeat(80).reshape(128,160).view(torch.float8_e4m3fn)
    path=Path(directory)/'source.safetensors'
    save_file({'table':source},path)
    with safe_open(path,framework='pt',device='cpu') as reader:
        mapped=reader.get_tensor('table')
    del reader
    torch.manual_seed(1831)
    ids=torch.randint(-5,133,(256,16))
    ids[0,:8]=torch.tensor([-1,0,31,32,63,64,127,128])
    expected=source.view(torch.uint8)[ids.clamp(0,127)]
    expected[(ids<0)|(ids>=128)]=0
    for threads in (1,4):
        table=CheckpointShards(160,source.dtype,directory,threads=threads)
        for start,end in ((64,128),(0,32),(32,64)):
            table.add(start,mapped[start:end])
        output=torch.empty(256,16,160,dtype=torch.uint8)
        table.gather_into(ids,output,128)
        assert torch.equal(output,expected)
    # Row cache: identical bytes through hits, duplicate misses and evictions
    # (a 64-row cache over a 128-row table), with buffered and O_DIRECT reads.
    import os
    for io_mode in ('buffered','direct'):
        with patch.dict(os.environ,{'FLASHNEXT_PLE_ROW_CACHE_GIB':str(64*168/2**30),'FLASHNEXT_PLE_IO':io_mode}):
            table=CheckpointShards(160,source.dtype,directory)
        for start,end in ((64,128),(0,32),(32,64)):
            table.add(start,mapped[start:end])
        for trial in range(20):
            step=torch.randint(-5,133,(8*4,16))
            step[1]=step[0]  # duplicates within one call
            want=source.view(torch.uint8)[step.clamp(0,127)]
            want[(step<0)|(step>=128)]=0
            output=torch.empty(32,16,160,dtype=torch.uint8)
            table.gather_into(step,output,128)
            assert torch.equal(output,want),(io_mode,trial)
        hits,misses,unique=table.row_cache_stats()
        assert hits>0 and unique<misses,(hits,misses,unique)
    for parts in ([(1,mapped)],[(0,mapped[:64]),(63,mapped[64:])]):
        table=CheckpointShards(160,source.dtype,directory)
        try:
            for start,value in parts: table.add(start,value)
            table.seal(128)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid coverage accepted')
    try:
        CheckpointShards(160,source.dtype,directory).add(0,source)
    except ValueError:
        pass
    else:
        raise AssertionError('Heap source accepted')
    # The real checkpoint places many tensors in one file. Advice for the
    # first tensor splits its VMA inside the following tensor's first page.
    adjacent_path=Path(directory)/'adjacent.safetensors'
    save_file({'a':source,'b':source.flip(0)},adjacent_path)
    adjacent=CheckpointShards(160,source.dtype,directory)
    with safe_open(adjacent_path,framework='pt',device='cpu') as reader:
        adjacent.add(0,reader.get_tensor('a'))
        adjacent.add(128,reader.get_tensor('b'))
    adjacent_ids=torch.arange(256,dtype=torch.int64)
    adjacent_output=torch.empty(256,160,dtype=torch.uint8)
    adjacent.gather_into(adjacent_ids,adjacent_output,256)
    assert torch.equal(adjacent_output,torch.cat([source,source.flip(0)]).view(torch.uint8))
    # Mock only distributed rank metadata; exercise the real parameter class.
    with patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank',return_value=0), \
         patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size',return_value=1):
        parameter=ModelWeightParameter(data=torch.empty(1,dtype=source.dtype).expand(128,160),
            input_dim=1,output_dim=0,weight_loader=lambda *args:None)
    assert parameter.untyped_storage().nbytes()==1
    print(json.dumps({'passed':True,'all_byte_patterns':True,'threads':[1,4],
        'reader_lifetime':True,'row_cache_buffered_and_direct':True,'gap_overlap_heap_refused':True,'split_vma_same_file':True,
        'real_parameter_class_with_mocked_tp_metadata':True,'parameter_storage_bytes':1}))
