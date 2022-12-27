# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import abc
from abc import ABC, abstractmethod

import torch
from _typeshed import Incomplete
from torch.ao.quantization.observer import (
    FixedQParamsObserver as FixedQParamsObserver,
)
from torch.ao.quantization.observer import (
    HistogramObserver as HistogramObserver,
)


        MovingAverageMinMaxObserver as MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver as \
        MovingAveragePerChannelMinMaxObserver,
        default_fixed_qparams_range_0to1_observer as \
        default_fixed_qparams_range_0to1_observer,
        default_fixed_qparams_range_neg1to1_observer as \
        default_fixed_qparams_range_neg1to1_observer
from typing import Any, Tuple

from torch.nn import Module as Module


class FakeQuantizeBase(ABC, Module, metaclass=abc.ABCMeta):
    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor
    def __init__(self) -> None: ...
    @abstractmethod
    def forward(self, x): ...
    @abstractmethod
    def calculate_qparams(self, **kwargs): ...
    def enable_fake_quant(self, enabled: bool = ...) -> None: ...
    def disable_fake_quant(self) -> None: ...
    def enable_observer(self, enabled: bool = ...) -> None: ...
    def disable_observer(self) -> None: ...
    with_args: Incomplete


class FakeQuantize(FakeQuantizeBase):
    scale: torch.Tensor
    zero_point: torch.Tensor
    activation_post_process: Incomplete
    quant_min: Incomplete
    quant_max: Incomplete
    dtype: Incomplete
    qscheme: Incomplete
    ch_axis: Incomplete
    is_per_channel: Incomplete

    def __init__(
        self, observer=..., quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ..., **observer_kwargs) -> None: ...

    def calculate_qparams(self): ...
    def forward(self, X): ...
    def extra_repr(self): ...


class FixedQParamsFakeQuantize(FakeQuantize):
    scale: Incomplete
    zero_point: Incomplete
    def __init__(self, observer) -> None: ...
    def calculate_qparams(self): ...
    def extra_repr(self): ...


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    is_symmetric_quant: Incomplete

    def __init__(
        self, observer: Any = ..., quant_min: int = ...,
        quant_max: int = ..., **observer_kwargs: Any) -> None: ...

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def extra_repr(self) -> str: ...
    def forward(self, X: torch.Tensor) -> torch.Tensor: ...


default_fake_quant: Incomplete
default_weight_fake_quant: Incomplete
default_dynamic_fake_quant: Incomplete
default_fixed_qparams_range_neg1to1_fake_quant: Incomplete
default_fixed_qparams_range_0to1_fake_quant: Incomplete
default_symmetric_fixed_qparams_fake_quant = \
        default_fixed_qparams_range_neg1to1_fake_quant
default_affine_fixed_qparams_fake_quant = \
        default_fixed_qparams_range_0to1_fake_quant
default_per_channel_weight_fake_quant: Incomplete
default_embedding_fake_quant: Incomplete
default_embedding_fake_quant_4bit: Incomplete
default_histogram_fake_quant: Incomplete
default_fused_act_fake_quant: Incomplete
default_fused_wt_fake_quant: Incomplete
default_fused_per_channel_wt_fake_quant: Incomplete
fused_wt_fake_quant_range_neg_127_to_127: Incomplete
fused_per_channel_wt_fake_quant_range_neg_127_to_127: Incomplete


def disable_fake_quant(mod) -> None: ...


def enable_fake_quant(mod) -> None: ...


def disable_observer(mod) -> None: ...


def enable_observer(mod) -> None: ...
