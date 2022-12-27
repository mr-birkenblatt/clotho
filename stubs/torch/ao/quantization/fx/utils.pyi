# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.ao.quantization.utils import is_per_channel as is_per_channel
from torch.ao.quantization.utils import is_per_tensor as is_per_tensor
from torch.fx import GraphModule as GraphModule
from torch.fx import map_arg as map_arg
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node


WEIGHT_INDEX_DICT: Incomplete
NON_QUANTIZABLE_WEIGHT_OPS: Incomplete
BIAS_INDEX_DICT: Incomplete


def graph_pretty_str(g, shorten: bool = ...) -> str: ...


def get_per_tensor_qparams(activation_post_process): ...


def get_quantize_node_info(
    activation_post_process: Callable) -> Optional[Tuple[str, Union[
                        Callable, str], Dict[str, Any]]]: ...


def quantize_node(
    in_node: Node, obs_module: torch.nn.Module, obs_node: Node,
    modules: Dict[str, torch.nn.Module], quantized_graph: Graph,
    node_name_to_scope: Dict[str, Tuple[str, type]], is_input: bool,
    output_prefix: str = ...) -> Node: ...


def get_custom_module_class_keys(
    custom_config_dict, custom_config_dict_key) -> List[Any]: ...


def get_linear_prepack_op_for_dtype(dtype): ...


def get_qconv_prepack_op(conv_op: Callable) -> Callable: ...


def get_qconv_op(conv_op: Callable, has_relu: bool) -> Callable: ...


def get_new_attr_name_with_prefix(prefix: str) -> Callable: ...


def collect_producer_nodes(node: Node) -> Optional[List[Node]]: ...


def graph_module_from_producer_nodes(
    root: GraphModule, producer_nodes: List[Node]) -> GraphModule: ...


def assert_and_get_unique_device(module: torch.nn.Module) -> Any: ...


def create_getattr_from_value(
    module: torch.nn.Module, graph: Graph, prefix: str,
    value: Any) -> Node: ...


def create_qparam_nodes(
    node_name: str, scale: Any, zero_point: Any, modules: Dict[str,
            torch.nn.Module], quantized_graph: Graph,
    node_name_to_scope: Dict[str, Tuple[str, type]]) -> Tuple[Node, Node]: ...


def all_node_args_have_no_tensors(
    node: Node, modules: Dict[str, torch.nn.Module], cache: Dict[Node,
            bool]) -> bool: ...


def all_node_args_except_first(node: Node) -> List[int]: ...


def return_arg_list(arg_indices: List[int]) -> Callable[[Node], List[int]]: ...


class NodeInfo(NamedTuple):
    op: Incomplete
    target: Incomplete
NON_OBSERVABLE_ARG_DICT: Dict[NodeInfo, Dict[Union[type, torch.dtype],
                    Callable[[Node], List[int]]]]
EMPTY_ARG_DICT: Dict[Union[type, torch.dtype], Callable[[Node], List[int]]]


def get_non_observable_arg_indexes_and_types(
    node: Node) -> Dict[Union[type, torch.dtype], Callable[[Node], List[
                        int]]]: ...


def node_return_type_is_int(node: Node) -> bool: ...


def is_get_tensor_info_node(node: Node) -> bool: ...


def maybe_get_next_module(
    node: Node, modules: Dict[str, nn.Module],
    target_module_type: Optional[Type[nn.Module]] = ...,
    target_functional_type: Any = ...) -> Optional[Node]: ...


def create_node_from_old_node_preserve_meta(
    quantized_graph: Graph, create_node_args: Tuple[Any, ...],
    old_node: Node) -> Node: ...
