from typing import Dict, List, Optional, Set

import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.fx.operator_schemas import (
    get_signature_for_torch_op as get_signature_for_torch_op,
)


aten: Incomplete
decomposition_table: Dict[str, torch.jit.ScriptFunction]
function_name_set: Set[str]

def check_decomposition_has_type_annotations(f) -> None: ...
def signatures_match(decomposition_sig, torch_op_sig): ...
def register_decomposition(aten_op, registry: Incomplete | None = ...): ...
def var_decomposition(input: Tensor, dim: Optional[List[int]] = ..., correction: Optional[int] = ..., keepdim: bool = ...) -> Tensor: ...
def var(input: Tensor, unbiased: bool = ...) -> Tensor: ...
