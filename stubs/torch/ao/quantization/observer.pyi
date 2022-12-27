# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import abc
from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.ao.quantization.utils import (
    calculate_qmin_qmax as calculate_qmin_qmax,
)
from torch.ao.quantization.utils import (
    check_min_max_valid as check_min_max_valid,
)


class _PartialWrapper:
    p: Incomplete
    callable_args: Incomplete
    def __init__(self, p) -> None: ...
    def __call__(self, *args, **keywords): ...
    def with_args(self, **kwargs): ...
    def with_callable_args(self, **kwargs): ...


ABC: Any


class ObserverBase(ABC, nn.Module, metaclass=abc.ABCMeta):
    dtype: Incomplete
    def __init__(self, dtype) -> None: ...
    @abstractmethod
    def forward(self, x): ...
    @abstractmethod
    def calculate_qparams(self, **kwargs): ...
    with_args: Incomplete
    with_callable_args: Incomplete


class UniformQuantizationObserverBase(ObserverBase, metaclass=abc.ABCMeta):
    eps: torch.Tensor
    qscheme: Incomplete
    reduce_range: Incomplete
    has_customized_qrange: Incomplete

    def __init__(
        self, dtype=..., qscheme=..., reduce_range: bool = ...,
        quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ...,
        factory_kwargs: Incomplete | None = ..., eps=...) -> None: ...

    def reset_min_max_vals(self) -> None: ...


class MinMaxObserver(UniformQuantizationObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self, dtype=..., qscheme=..., reduce_range: bool = ...,
        quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ...,
        factory_kwargs: Incomplete | None = ..., eps=...) -> None: ...

    def forward(self, x_orig): ...
    def calculate_qparams(self): ...
    def extra_repr(self): ...
    def reset_min_max_vals(self) -> None: ...


class MovingAverageMinMaxObserver(MinMaxObserver):
    averaging_constant: Incomplete

    def __init__(
        self, averaging_constant: float = ..., dtype=..., qscheme=...,
        reduce_range: bool = ..., quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ..., eps=..., **kwargs) -> None: ...

    def forward(self, x_orig): ...


class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor
    ch_axis: Incomplete

    def __init__(
        self, ch_axis: int = ..., dtype=..., qscheme=...,
        reduce_range: bool = ..., quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ...,
        factory_kwargs: Incomplete | None = ..., eps=...) -> None: ...

    def forward(self, x_orig): ...
    def calculate_qparams(self): ...
    def extra_repr(self): ...
    def reset_min_max_vals(self) -> None: ...


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    averaging_constant: Incomplete

    def __init__(
        self, averaging_constant: float = ..., ch_axis: int = ..., dtype=...,
        qscheme=..., reduce_range: bool = ...,
        quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ..., eps=..., **kwargs) -> None: ...

    def forward(self, x_orig): ...


class HistogramObserver(UniformQuantizationObserverBase):
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    bins: Incomplete
    dst_nbins: Incomplete
    upsample_rate: Incomplete

    def __init__(
        self, bins: int = ..., upsample_rate: int = ...,
        dtype: torch.dtype = ..., qscheme=..., reduce_range: bool = ...,
        quant_min: Incomplete | None = ...,
        quant_max: Incomplete | None = ...,
        factory_kwargs: Incomplete | None = ..., eps=...) -> None: ...

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor: ...
    def calculate_qparams(self): ...


class FixedQParamsObserver(ObserverBase):
    scale: torch.Tensor
    zero_point: torch.Tensor
    quant_min: Incomplete
    quant_max: Incomplete
    dtype: Incomplete
    qscheme: Incomplete

    def __init__(
        self, scale, zero_point, dtype=..., qscheme=...,
        quant_min: int = ..., quant_max: int = ...) -> None: ...

    def forward(self, X): ...
    def calculate_qparams(self): ...


class PlaceholderObserver(ObserverBase):
    dtype: Incomplete
    custom_op: Incomplete
    compute_dtype: Incomplete

    def __init__(
        self, dtype=..., custom_op_name: str = ...,
        compute_dtype: Incomplete | None = ...) -> None: ...

    def forward(self, x): ...
    def calculate_qparams(self) -> None: ...


class RecordingObserver(ObserverBase):
    __annotations__: Incomplete
    tensor_val: Incomplete
    def __init__(self, dtype=..., **kwargs) -> None: ...
    def forward(self, x): ...
    def calculate_qparams(self) -> None: ...
    def get_tensor_value(self): ...


class NoopObserver(ObserverBase):
    dtype: Incomplete
    custom_op: Incomplete
    def __init__(self, dtype=..., custom_op_name: str = ...) -> None: ...
    def forward(self, x): ...
    def calculate_qparams(self) -> None: ...


class ReuseInputObserver(ObserverBase):
    def __init__(self) -> None: ...
    def forward(self, x): ...
    def calculate_qparams(self) -> None: ...


def get_observer_state_dict(mod): ...


def load_observer_state_dict(mod, obs_dict) -> None: ...


default_observer: Incomplete
default_placeholder_observer = PlaceholderObserver
default_debug_observer = RecordingObserver
default_weight_observer: Incomplete
weight_observer_range_neg_127_to_127: Incomplete
default_histogram_observer: Incomplete
default_per_channel_weight_observer: Incomplete
per_channel_weight_observer_range_neg_127_to_127: Incomplete
default_dynamic_quant_observer: Incomplete
default_float_qparams_observer: Incomplete
default_float_qparams_observer_4bit: Incomplete
default_fixed_qparams_range_neg1to1_observer: Incomplete
default_fixed_qparams_range_0to1_observer: Incomplete
default_symmetric_fixed_qparams_observer = \
    default_fixed_qparams_range_neg1to1_observer
default_affine_fixed_qparams_observer = \
    default_fixed_qparams_range_0to1_observer
default_reuse_input_observer = ReuseInputObserver
