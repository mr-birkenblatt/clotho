# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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


def CompileSpec(
    inputs, outputs, backend=..., allow_low_precision: bool = ...): ...


def preprocess(
    script_module: torch._C.ScriptObject, compile_spec: Dict[str, Tuple]): ...
