# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node

from ..observer import ObserverBase as ObserverBase
from ..observer import PerChannelMinMaxObserver as PerChannelMinMaxObserver
from ..utils import check_min_max_valid as check_min_max_valid
from .utils import (
    get_new_attr_name_with_prefix as get_new_attr_name_with_prefix,
)
from .utils import maybe_get_next_module as maybe_get_next_module
from .utils import WEIGHT_INDEX_DICT as WEIGHT_INDEX_DICT


def reshape_scale(
    scale: torch.Tensor, axis: int, input: torch.Tensor) -> torch.Tensor: ...


class _InputEqualizationObserver(nn.Module):
    dtype: Incomplete
    qscheme: Incomplete
    input_obs: Incomplete
    equalization_scale: Incomplete
    equalization_shape: Incomplete

    def __init__(
        self, dtype=..., qscheme=..., quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ...,
        factory_kwargs: Incomplete | None = ...) -> None: ...

    def forward(self, x_orig): ...
    def get_input_minmax(self): ...
    def set_equalization_scale(self, equalization_scale) -> None: ...
    def calculate_scaled_minmax(self): ...
    with_args: Incomplete


class _WeightEqualizationObserver(nn.Module):
    dtype: Incomplete
    qscheme: Incomplete
    ch_axis: int
    weight_col_obs: Incomplete
    equalization_scale: Incomplete

    def __init__(
        self, dtype=..., qscheme=..., quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ...,
        factory_kwargs: Incomplete | None = ...) -> None: ...

    def forward(self, w_orig): ...
    def get_weight_col_minmax(self): ...
    def set_equalization_scale(self, equalization_scale) -> None: ...
    with_args: Incomplete


def calculate_equalization_scale(
    input_obs: _InputEqualizationObserver,
    weight_obs: _WeightEqualizationObserver) -> torch.Tensor: ...


class EqualizationQConfig:
    def __new__(cls, input_activation=..., weight=...): ...


input_equalization_observer: Incomplete
weight_equalization_observer: Incomplete
default_equalization_qconfig: Incomplete


def fused_module_supports_equalization(module) -> bool: ...


def nn_module_supports_equalization(module) -> bool: ...


def node_supports_equalization(node: Node, modules) -> bool: ...


def is_equalization_observer(observer: nn.Module) -> bool: ...


def get_op_node_and_weight_eq_obs(
    input_eq_obs_node: Node, model: GraphModule, modules: Dict[str,
            nn.Module]) -> Tuple[Optional[Node], Optional[
                _WeightEqualizationObserver]]: ...


def maybe_get_weight_eq_obs_node(
    op_node: Node, modules: Dict[str, nn.Module]) -> Optional[Node]: ...


def maybe_get_next_input_eq_obs(
    node: Node, modules: Dict[str, nn.Module]) -> Optional[
        _InputEqualizationObserver]: ...


def maybe_get_next_equalization_scale(
    node: Node, modules: Dict[str, nn.Module]) -> Optional[torch.Tensor]: ...


def scale_input_observer(
    node: Node, modules: Dict[str, nn.Module]) -> None: ...


def scale_weight_node(
    node: Node, modules: Dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: Optional[torch.Tensor]) -> None: ...


def scale_weight_functional(
    op_node: Node, model: GraphModule, modules: Dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: Optional[torch.Tensor]) -> None: ...


def clear_weight_quant_obs_node(
    op_node: Node, modules: Dict[str, nn.Module]) -> None: ...


def remove_node(model: GraphModule, node: Node, prev_node: Node): ...


def update_obs_for_equalization(
    model: GraphModule, modules: Dict[str, nn.Module]) -> Dict[
        str, _WeightEqualizationObserver]: ...


def convert_eq_obs(
    model: GraphModule, modules: Dict[str, nn.Module],
    weight_eq_obs_dict: Dict[str, _WeightEqualizationObserver]) -> None: ...


def get_layer_sqnr_dict(
    model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> Dict[
        str, float]: ...


def get_equalization_qconfig_dict(
    layer_sqnr_dict: Dict[str, float], num_layers_to_equalize: int) -> Any: ...
