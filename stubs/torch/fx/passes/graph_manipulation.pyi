# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.graph import Graph as Graph
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.node import Argument as Argument
from torch.fx.node import Node as Node


        Target as Target, map_aggregate as map_aggregate, map_arg as map_arg
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from torch.fx.passes.param_fetch import (
    lift_lowering_attrs_to_nodes as lift_lowering_attrs_to_nodes,
)
from torch.fx.passes.shape_prop import ShapeProp as ShapeProp


def replace_target_nodes_with(
    fx_module: GraphModule, old_op: str, old_target: Target, new_op: str,
    new_target: Target): ...


class size_bytes(NamedTuple):
    output_size: int
    total_size: int


def get_size_of_all_nodes(
    fx_module: GraphModule,
    args: Optional[List[torch.Tensor]] = ...) -> None: ...


def get_tensor_meta(node: Node) -> Any: ...


def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes: ...


def serialize_shape(shape: torch.Size) -> str: ...


def serialize_stride(stride: Tuple[int]) -> str: ...


def serialize_tensor_quantization(
    tensor: torch.Tensor, weights: Dict, pcq_prefix: str) -> Tuple[Dict,
        Dict]: ...


def serialize_weight(
    tensor: torch.Tensor, weights: Dict, name: str) -> Dict: ...


def serialize_leaf_module(
    node: Node, weights_metadata: Dict, weights: Dict,
    name_prefix: str) -> Dict: ...


def serialize_module(
    fx_module: GraphModule, weights: Dict, name_prefix: str = ...) -> Dict: ...
