"""CPU configuration/weight-header preflight, without loading target weights."""
import argparse
import json
from pathlib import Path

from safetensors import safe_open
from dflash_epoch7 import checkpoint_shapes
from vllm import ModelRegistry
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--draft", required=True)
a = p.parse_args()
ModelRegistry.register_model("DFlashQwen3DSparkModel", "dflash_epoch7:DFlashQwen3DSparkModel")
seen = {}
for path in Path(a.draft).glob("*.safetensors"):
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key in seen:
                raise ValueError(f"Duplicate tensor {key}")
            tensor = f.get_slice(key)
            seen[key] = (tuple(tensor.get_shape()), tensor.get_dtype())
expected = checkpoint_shapes()
assert set(seen) == set(expected)
assert all(seen[k] == (shape, "BF16") for k, shape in expected.items())
model = ModelConfig(model=a.model, max_model_len=212992, enforce_eager=True)
configs = []
for k in (4, 5, 7):
    spec = SpeculativeConfig(target_model_config=model,
        target_parallel_config=ParallelConfig(), model=a.draft, method="dflash",
        num_speculative_tokens=k, quantization=None, kv_cache_dtype="auto")
    assert spec.draft_model_config.quantization is None
    assert str(spec.draft_model_config.dtype) == "torch.bfloat16"
    assert spec.max_num_new_slots_for_drafting == k - 1
    assert tuple(get_eagle3_aux_layers_from_config(spec)) == (4, 16, 24, 36, 44)
    assert spec.draft_model_config.architectures == ["DFlashQwen3DSparkModel"]
    configs.append({"K": k, "additional_slots": spec.max_num_new_slots_for_drafting})
print(json.dumps({"checkpoint_tensors": len(seen), "configs": configs,
                  "status": "config_passed_no_inference"}, indent=2))
