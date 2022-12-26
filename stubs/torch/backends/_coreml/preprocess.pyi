from typing import Dict, Tuple

import torch
from _typeshed import Incomplete


CT_METADATA_VERSION: str
CT_METADATA_SOURCE: str

class ScalarType:
    Float: int
    Double: int
    Int: int
    Long: int
    Undefined: int

torch_to_mil_types: Incomplete

class CoreMLComputeUnit:
    CPU: str
    CPUAndGPU: str
    ALL: str

def TensorSpec(shape, dtype=...): ...
def CompileSpec(inputs, outputs, backend=..., allow_low_precision: bool = ...): ...
def preprocess(script_module: torch._C.ScriptObject, compile_spec: Dict[str, Tuple]): ...
