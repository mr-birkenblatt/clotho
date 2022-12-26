from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch
from torch._jit_internal import boolean_dispatched as boolean_dispatched
from torch._ops import OpOverload as OpOverload
from torch._ops import OpOverloadPacket as OpOverloadPacket

from ._compatibility import compatibility as compatibility
from .node import Argument as Argument


class ArgsKwargsPair(NamedTuple):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class _FakeGlobalNamespace:
    def __getattr__(self, name): ...

def check_for_mutable_operation(target: Callable, args: Tuple['Argument', ...], kwargs: Dict[str, 'Argument']): ...
def get_signature_for_torch_op(op: Callable, return_schemas: bool = ...): ...
def create_type_hint(x): ...
def type_matches(signature_type: Any, argument_type: Any): ...
def normalize_function(target: Callable, args: Tuple[Any], kwargs: Optional[Dict[str, Any]] = ..., arg_types: Optional[Tuple[Any]] = ..., kwarg_types: Optional[Dict[str, Any]] = ..., normalize_to_only_use_kwargs: bool = ...) -> Optional[ArgsKwargsPair]: ...
def normalize_module(root: torch.nn.Module, target: str, args: Tuple[Any], kwargs: Optional[Dict[str, Any]] = ..., normalize_to_only_use_kwargs: bool = ...) -> Optional[ArgsKwargsPair]: ...
