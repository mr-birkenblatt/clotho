# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, Dict, List, Set

import torch
from _typeshed import Incomplete
from torch import nn
from torch.ao.quantization import prepare as prepare
from torch.ao.quantization.quantization_mappings import (
    get_default_compare_output_module_list as get_default_compare_output_module_list,
)


NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST: Incomplete


def compare_weights(
    float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]) -> Dict[
        str, Dict[str, torch.Tensor]]: ...


def get_logger_dict(mod: nn.Module, prefix: str = ...) -> Dict[str, Dict]: ...


class Logger(nn.Module):
    stats: Incomplete
    dtype: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x) -> None: ...


class ShadowLogger(Logger):
    def __init__(self) -> None: ...
    def forward(self, x, y) -> None: ...


class OutputLogger(Logger):
    def __init__(self) -> None: ...
    def forward(self, x): ...


class Shadow(nn.Module):
    orig_module: Incomplete
    shadow_module: Incomplete
    dequant: Incomplete
    logger: Incomplete
    def __init__(self, q_module, float_module, logger_cls) -> None: ...
    def forward(self, *x) -> torch.Tensor: ...
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor: ...
    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor: ...
    def cat(self, x: List[torch.Tensor], dim: int = ...) -> torch.Tensor: ...
    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


def prepare_model_with_stubs(
    float_module: nn.Module, q_module: nn.Module,
    module_swap_list: Set[type], logger_cls: Callable) -> None: ...


def compare_model_stub(
    float_model: nn.Module, q_model: nn.Module, module_swap_list: Set[type],
    *data, logger_cls=...) -> Dict[str, Dict]: ...


def get_matching_activations(
    float_module: nn.Module, q_module: nn.Module) -> Dict[str, Dict[
                str, torch.Tensor]]: ...


def prepare_model_outputs(
    float_module: nn.Module, q_module: nn.Module, logger_cls=...,
    allow_list: Incomplete | None = ...) -> None: ...


def compare_model_outputs(
    float_model: nn.Module, q_model: nn.Module, *data, logger_cls=...,
    allow_list: Incomplete | None = ...) -> Dict[str, Dict[
                str, torch.Tensor]]: ...
