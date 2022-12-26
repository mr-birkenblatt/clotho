from typing import Any, Callable, Dict, Optional, Tuple

import torch.fx
from _typeshed import Incomplete
from torch.fx import Proxy as Proxy
from torch.fx import Transformer as Transformer
from torch.fx.node import Argument as Argument
from torch.fx.node import map_aggregate as map_aggregate
from torch.fx.node import Node as Node
from torch.fx.node import Target as Target
from torch.fx.operator_schemas import create_type_hint as create_type_hint
from torch.fx.operator_schemas import normalize_function as normalize_function
from torch.fx.operator_schemas import normalize_module as normalize_module

from .schema_type_annotation import (
    AnnotateTypesWithSchema as AnnotateTypesWithSchema,
)


class NormalizeArgs(Transformer):
    node_map: Incomplete
    normalize_to_only_use_kwargs: Incomplete
    def __init__(self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool = ...) -> None: ...
    def run_node(self, n: Node) -> Any: ...
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any], arg_types: Optional[Tuple[Any, ...]] = ..., kwarg_types: Optional[Dict[str, Any]] = ...): ...
    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]): ...

class NormalizeOperators(AnnotateTypesWithSchema):
    binary_magic_method_remap: Dict[Callable[[Any, Any], Any], Callable[[Any, Any], Any]]
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]): ...