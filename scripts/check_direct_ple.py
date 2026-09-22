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
    # Mock only distributed rank metadata; exercise the real parameter class.
    with patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank',return_value=0), \
         patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size',return_value=1):
        parameter=ModelWeightParameter(data=torch.empty(1,dtype=source.dtype).expand(128,160),
            input_dim=1,output_dim=0,weight_loader=lambda *args:None)
    assert parameter.untyped_storage().nbytes()==1
    print(json.dumps({'passed':True,'all_byte_patterns':True,'threads':[1,4],
        'reader_lifetime':True,'gap_overlap_heap_refused':True,
        'real_parameter_class_with_mocked_tp_metadata':True,'parameter_storage_bytes':1}))
