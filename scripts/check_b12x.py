"""Run pinned upstream B12x numerical and changing CUDA graph checks.

Supply the tests directory from the pinned vLLM source in --vllm-source.
The optional b12x package can live on PYTHONPATH without changing the runtime.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--vllm-source',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
if Path(a.output).exists():
    p.error('output already exists')
sys.path.insert(0,a.vllm_source)

import torch
import vllm
from tests.kernels.moe.test_b12x import (
    MoEActivation, _has_b12x_moe, _make_b12x_moe_case,
    _make_b12x_moe_kernel, fused_topk, test_b12x_moe_matches_torch,
)
from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
from vllm.v1.worker.workspace import init_workspace_manager, lock_workspace, reset_workspace_manager

assert vllm.__version__ == '0.29.1rc1.dev518+ga33b3bac5'
assert _has_b12x_moe(), 'B12x MoE unavailable on this device'
init_workspace_manager(torch.device('cuda'))
started=time.monotonic()
test_b12x_moe_matches_torch('nvfp4','nvfp4',MoEActivation.SILU,None)
rows=[]
with torch.inference_mode(), set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))):
    for tokens in (1,8,24):
        reset_workspace_manager()
        init_workspace_manager(torch.device('cuda'))
        case=_make_b12x_moe_case('nvfp4','nvfp4',tokens=tokens,seed=23+tokens)
        kernel=_make_b12x_moe_kernel(case.hidden_states,case.w1,case.w2,case.topk,
                                     case.activation,case.quant_config)
        tw,ti,_=fused_topk(case.hidden_states,case.score,case.topk,renormalize=False)
        def apply():
            return kernel.apply(hidden_states=case.hidden_states,w1=case.w1,w2=case.w2,
                topk_weights=tw,topk_ids=ti,activation=case.activation,
                global_num_experts=case.w1.shape[0],expert_map=None,
                apply_router_weight_on_input=False)
        for _ in range(3): apply()
        torch.cuda.synchronize()
        lock_workspace()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=torch.cuda.Stream()):
            actual=apply()
        maximum_error=0.0
        for replay in range(16):
            case.hidden_states.normal_(std=0.1)
            case.score.normal_()
            next_tw,next_ti,_=fused_topk(case.hidden_states,case.score,case.topk,renormalize=False)
            tw.copy_(next_tw); ti.copy_(next_ti)
            expected=apply().clone()
            graph.replay(); torch.cuda.synchronize()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual,expected,atol=2e-2,rtol=2e-2)
            maximum_error=max(maximum_error,(actual.float()-expected.float()).abs().max().item())
        rows.append({'tokens':tokens,'changing_replays':16,'maximum_absolute_error':maximum_error})
reset_workspace_manager()
report={'passed':True,'scope':'small upstream MoE fixture, not full-model quality or throughput',
        'vllm':vllm.__version__,'b12x':importlib.metadata.version('b12x'),
        'cutlass_dsl':importlib.metadata.version('nvidia-cutlass-dsl'),
        'upstream_numerical_reference':True,'graph_checks':rows,'seconds':time.monotonic()-started}
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with Path(a.output).open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
