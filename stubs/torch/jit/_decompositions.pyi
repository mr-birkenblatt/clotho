# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.fx.operator_schemas import as, get_signature_for_torch_op


        get_signature_for_torch_op
from typing import Dict, List, Optional, Set


aten: Incomplete
decomposition_table: Dict[str, torch.jit.ScriptFunction]
function_name_set: Set[str]


def check_decomposition_has_type_annotations(f) -> None: ...


def signatures_match(decomposition_sig, torch_op_sig): ...


def register_decomposition(aten_op, registry: Incomplete | None = ...): ...


def var_decomposition(
    input: Tensor, dim: Optional[List[int]] = ...,
    correction: Optional[int] = ..., keepdim: bool = ...) -> Tensor: ...


def var(input: Tensor, unbiased: bool = ...) -> Tensor: ...
