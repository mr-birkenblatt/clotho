# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from types import ModuleType
from typing import Any, Callable, Dict, Tuple

import torch.fx
from _typeshed import Incomplete

from .mappings import conv_ops as conv_ops
from .quantization_state import AutoQuantizationState as AutoQuantizationState
from .utils import (
    AutoQuantizationStateModuleDict as AutoQuantizationStateModuleDict,
)
from .utils import get_packable_arg_idxs as get_packable_arg_idxs


class AllModuleTracer(torch.fx.Tracer):
    node_name_to_dtype: Dict[str, Any]

    def __init__(
        self, autowrap_modules: Tuple[ModuleType] = ...,
        autowrap_functions: Tuple[Callable, ...] = ...,
        param_shapes_constant: bool = ...) -> None: ...

    def is_leaf_module(self, m, module_qualified_name) -> bool: ...

    def create_node(
        self, kind, target, args, kwargs, name: Incomplete | None = ...,
        type_expr: Incomplete | None = ...): ...

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any],
        args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any: ...


def rewrite_for_scripting(mod: torch.nn.Module) -> torch.nn.Module: ...
