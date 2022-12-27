# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from types import ModuleType
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from _typeshed import Incomplete
from torch._C import ScriptObject as ScriptObject

from ._compatibility import compatibility as compatibility
from .graph import Graph as Graph
from .graph_module import GraphModule as GraphModule
from .node import Argument as Argument
from .node import base_types as base_types
from .node import map_aggregate as map_aggregate
from .proxy import ParameterProxy as ParameterProxy
from .proxy import Proxy as Proxy
from .proxy import TracerBase as TracerBase


HAS_VARSTUFF: Incomplete


class ProxyableClassMeta(type):
    def __init__(cls, name, bases, attrs) -> None: ...
    def __call__(cls, *args, **kwargs): ...


class PHBase:
    ...


PH: Incomplete


class Tracer(TracerBase):
    param_shapes_constant: Incomplete
    submodule_paths: Incomplete

    def __init__(
        self, autowrap_modules: Tuple[ModuleType] = ...,
        autowrap_functions: Tuple[Callable, ...] = ...,
        param_shapes_constant: bool = ...) -> None: ...

    def create_arg(self, a: Any) -> Argument: ...

    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str) -> bool: ...

    def path_of_module(self, mod: torch.nn.Module) -> str: ...

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any],
        args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any: ...

    def create_args_for_root(
        self, root_fn, is_module, concrete_args: Incomplete | None = ...): ...

    root: Incomplete
    graph: Incomplete
    tensor_attrs: Incomplete

    def trace(
        self, root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = ...) -> Graph: ...


class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any
    def revert(self) -> None: ...


class _PatchedFnSetItem(_PatchedFn):
    def revert(self) -> None: ...


class _PatchedFnDel(_PatchedFn):
    def revert(self) -> None: ...


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self) -> None: ...


class _Patcher:
    patches_made: Incomplete
    visited: Incomplete
    def __init__(self) -> None: ...

    def patch(
        self, frame_dict: Dict[str, Any], name: str, new_fn: Callable,
        deduplicate: bool = ...): ...

    def patch_method(
        self, cls: type, name: str, new_fn: Callable,
        deduplicate: bool = ...): ...

    def visit_once(self, thing: Any): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...


def wrap(fn_or_name: Union[str, Callable]): ...


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = ...) -> GraphModule: ...
