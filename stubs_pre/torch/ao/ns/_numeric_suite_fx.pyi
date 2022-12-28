# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from _typeshed import Incomplete
from torch import nn
from torch.ao.ns.fx.graph_matcher import (
    get_matching_subgraph_pairs as get_matching_subgraph_pairs,
)
from torch.ao.ns.fx.graph_matcher import (
    get_type_a_related_to_b as get_type_a_related_to_b,
)
from torch.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops
from torch.ao.quantization import quantize_fx
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node

from .fx.graph_passes import add_loggers_to_model as add_loggers_to_model
from .fx.graph_passes import create_a_shadows_b as create_a_shadows_b
from .fx.ns_types import NSNodeTargetType as NSNodeTargetType
from .fx.ns_types import NSResultsType as NSResultsType
from .fx.ns_types import NSSingleResultValuesType as NSSingleResultValuesType
from .fx.utils import get_target_type_str as get_target_type_str
from .fx.utils import maybe_add_missing_fqns as maybe_add_missing_fqns
from .fx.utils import rekey_logger_info_on_node_name_of_model
from .fx.weight_utils import (
    extract_weight_from_node as extract_weight_from_node,
)


RNNReturnType: Incomplete


class OutputLogger(nn.Module):
    stats: List[torch.Tensor]
    stats_rnn: List[RNNReturnType]
    ref_node_name: Incomplete
    prev_node_name: Incomplete
    model_name: Incomplete
    ref_name: Incomplete
    prev_node_target_type: Incomplete
    ref_node_target_type: Incomplete
    results_type: Incomplete
    index_within_arg: Incomplete
    index_of_arg: Incomplete
    fqn: Incomplete

    def __init__(
        self, ref_node_name: str, prev_node_name: str, model_name: str,
        ref_name: str, prev_node_target_type: str, ref_node_target_type: str,
        results_type: str, index_within_arg: int, index_of_arg: int,
        fqn: Optional[str]) -> None: ...

    def forward(self, x): ...


class NSTracer(quantize_fx.QuantizationTracer):

    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str) -> bool: ...


def extract_weights(
    model_name_a: str, model_a: nn.Module, model_name_b: str,
    model_b: nn.Module,
    base_name_to_sets_of_related_ops: Optional[
        Dict[str, Set[NSNodeTargetType]]] = ...,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = ...,
    op_to_type_to_weight_extraction_fn: Optional[
        Dict[str, Dict[Callable, Callable]]] = ...) -> NSResultsType: ...


def add_loggers(
    name_a: str, model_a: nn.Module, name_b: str, model_b: nn.Module,
    logger_cls: Callable, should_log_inputs: bool = ...,
    base_name_to_sets_of_related_ops: Optional[
        Dict[str, Set[NSNodeTargetType]]] = ...,
    unmatchable_types_map: Optional[
        Dict[str, Set[NSNodeTargetType]]] = ...) -> Tuple[
        nn.Module, nn.Module]: ...


def extract_logger_info(
    model_a: nn.Module, model_b: nn.Module, logger_cls: Callable,
    model_name_to_use_for_layer_names: str) -> NSResultsType: ...


def add_shadow_loggers(
    name_a: str, model_a: nn.Module, name_b: str, model_b: nn.Module,
    logger_cls: Callable, should_log_inputs: bool = ...,
    base_name_to_sets_of_related_ops: Optional[
        Dict[str, Set[NSNodeTargetType]]] = ...,
    node_type_to_io_type_map: Optional[
        Dict[str, Set[NSNodeTargetType]]] = ...,
    unmatchable_types_map: Optional[
        Dict[str, Set[NSNodeTargetType]]] = ...) -> nn.Module: ...


def extract_shadow_logger_info(
    model_a_shadows_b: nn.Module, logger_cls: Callable,
    model_name_to_use_for_layer_names: str) -> NSResultsType: ...


def extend_logger_results_with_comparison(
    results: NSResultsType, model_name_1: str, model_name_2: str,
    comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    comparison_name: str) -> None: ...
