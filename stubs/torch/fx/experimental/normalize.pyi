# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch.fx

from .schema_type_annotation import AnnotateTypesWithSchema, as


        AnnotateTypesWithSchema
from _typeshed import Incomplete
from torch.fx import Proxy as Proxy
from torch.fx import Transformer as Transformer
from torch.fx.node import Argument as Argument
from torch.fx.node import Node as Node


        Target as Target, map_aggregate as map_aggregate
from torch.fx.operator_schemas import create_type_hint as create_type_hint


        normalize_function as normalize_function,
        normalize_module as normalize_module
from typing import Any, Callable, Dict, Optional, Tuple


class NormalizeArgs(Transformer):
    node_map: Incomplete
    normalize_to_only_use_kwargs: Incomplete

    def __init__(
        self, module: torch.fx.GraphModule,
        normalize_to_only_use_kwargs: bool = ...) -> None: ...

    def run_node(self, n: Node) -> Any: ...

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
        Any], arg_types: Optional[Tuple[Any, ...]] = ...,
        kwarg_types: Optional[Dict[str, Any]] = ...): ...

    def call_module(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
        Any]): ...


class NormalizeOperators(AnnotateTypesWithSchema):
    binary_magic_method_remap: Dict[Callable[[Any, Any], Any], Callable[[Any,
            Any], Any]]

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
        Any]): ...
