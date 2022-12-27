# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch

from .backend_config import (
    get_tensorrt_backend_config_dict as get_tensorrt_backend_config_dict,
)
from .fx import fuse as fuse
from .fx import prepare as prepare
from .fx.convert import convert as convert
from .fx.graph_module import ObservedGraphModule as ObservedGraphModule
from .fx.qconfig_utils import (
    check_is_valid_convert_custom_config_dict as check_is_valid_convert_custom_config_dict,
)


        check_is_valid_fuse_custom_config_dict as \
        check_is_valid_fuse_custom_config_dict,
        check_is_valid_prepare_custom_config_dict as \
        check_is_valid_prepare_custom_config_dict,
        check_is_valid_qconfig_dict as check_is_valid_qconfig_dict
from typing import Any, Callable, Dict, List, Optional, Tuple

from _typeshed import Incomplete
from torch.fx import GraphModule as GraphModule
from torch.fx._symbolic_trace import Tracer as Tracer
from torch.fx.node import Argument as Argument
from torch.fx.node import Node as Node
from torch.fx.node import Target as Target

from .fx.utils import (
    get_custom_module_class_keys as get_custom_module_class_keys,
)
from .fx.utils import graph_pretty_str as graph_pretty_str


class Scope:
    module_path: Incomplete
    module_type: Incomplete
    def __init__(self, module_path: str, module_type: Any) -> None: ...


class ScopeContextManager:
    prev_module_type: Incomplete
    prev_module_path: Incomplete
    scope: Incomplete

    def __init__(
        self, scope: Scope, current_module: torch.nn.Module,
        current_module_path: str) -> None: ...

    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...


class QuantizationTracer(Tracer):
    skipped_module_names: Incomplete
    skipped_module_classes: Incomplete
    scope: Incomplete
    node_name_to_scope: Incomplete
    record_stack_traces: bool

    def __init__(
        self, skipped_module_names: List[str],
        skipped_module_classes: List[Callable]) -> None: ...

    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str) -> bool: ...

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any],
        args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any: ...

    def create_node(
        self, kind: str, target: Target, args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument], name: Optional[str] = ...,
        type_expr: Optional[Any] = ...) -> Node: ...


def fuse_fx(
    model: torch.nn.Module, fuse_custom_config_dict: Optional[Dict[str,
    Any]] = ..., backend_config_dict: Optional[Dict[str,
    Any]] = ...) -> GraphModule: ...


def prepare_fx(
    model: torch.nn.Module, qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = ...,
    equalization_qconfig_dict: Optional[Dict[str, Any]] = ...,
    backend_config_dict: Optional[Dict[str,
    Any]] = ...) -> ObservedGraphModule: ...


def prepare_qat_fx(
    model: torch.nn.Module, qconfig_dict: Any,
    prepare_custom_config_dict: Optional[Dict[str, Any]] = ...,
    backend_config_dict: Optional[Dict[str,
    Any]] = ...) -> ObservedGraphModule: ...


def convert_fx(
    graph_module: GraphModule, is_reference: bool = ...,
    convert_custom_config_dict: Optional[Dict[str, Any]] = ...,
    _remove_qconfig: bool = ..., qconfig_dict: Dict[str, Any] = ...,
    backend_config_dict: Dict[str, Any] = ...) -> torch.nn.Module: ...
