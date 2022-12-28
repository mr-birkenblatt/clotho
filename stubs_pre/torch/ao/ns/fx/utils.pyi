# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from _typeshed import Incomplete
from torch.ao.quantization import FakeQuantizeBase as FakeQuantizeBase
from torch.ao.quantization import ObserverBase as ObserverBase
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node

from .ns_types import NSNodeTargetType as NSNodeTargetType
from .ns_types import NSResultsType as NSResultsType


toq: Incomplete


class NodeInputOrOutputType(enum.Enum):
    FP32: Incomplete
    INT8: Incomplete
    FP16: Incomplete
    UNKNOWN: Incomplete
    FP32_OR_INT8: Incomplete


def get_node_first_input_and_output_type(
    node: Node, gm: GraphModule, logger_cls: Callable,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]]) -> Tuple[
        NodeInputOrOutputType, NodeInputOrOutputType]: ...


def get_node_input_qparams(
    node: Node, gm: GraphModule,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
    ) -> Optional[Tuple[Union[
        torch.Tensor, float], Union[torch.Tensor, int]]]: ...


def return_first_non_observer_node(node: Node, gm: GraphModule) -> Node: ...


def get_number_of_non_param_args(node: Node, gm: GraphModule) -> int: ...


def get_arg_indices_of_inputs_to_log(node: Node) -> List[int]: ...


def get_target_type_str(node: Node, gm: GraphModule) -> str: ...


def rekey_logger_info_on_node_name_of_model(
    results: NSResultsType, model_name: str) -> NSResultsType: ...


def maybe_add_missing_fqns(results: NSResultsType) -> None: ...


def maybe_dequantize_first_two_tensor_args_and_handle_tuples(f): ...


def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


def compute_normalized_l2_error(
    x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


def compute_cosine_similarity(
    x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


def op_type_supports_shadowing(node: Node) -> bool: ...
