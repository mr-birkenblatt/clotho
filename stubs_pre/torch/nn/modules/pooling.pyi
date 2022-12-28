# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..common_types import (
    _ratio_2_t,
    _ratio_3_t,
    _size_1_t,
    _size_2_opt_t,
    _size_2_t,
    _size_3_opt_t,
    _size_3_t,
    _size_any_opt_t,
    _size_any_t,
)
from .module import Module as Module


class _MaxPoolNd(Module):
    __constants__: Incomplete
    return_indices: bool
    ceil_mode: bool
    kernel_size: Incomplete
    stride: Incomplete
    padding: Incomplete
    dilation: Incomplete

    def __init__(
        self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = ...,
        padding: _size_any_t = ..., dilation: _size_any_t = ...,
        return_indices: bool = ..., ceil_mode: bool = ...) -> None: ...

    def extra_repr(self) -> str: ...


class MaxPool1d(_MaxPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t
    def forward(self, input: Tensor): ...


class MaxPool2d(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t
    def forward(self, input: Tensor): ...


class MaxPool3d(_MaxPoolNd):
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t
    def forward(self, input: Tensor): ...


class _MaxUnpoolNd(Module):
    def extra_repr(self) -> str: ...


class MaxUnpool1d(_MaxUnpoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t

    def __init__(
        self, kernel_size: _size_1_t, stride: Optional[_size_1_t] = ...,
        padding: _size_1_t = ...) -> None: ...

    def forward(
        self, input: Tensor, indices: Tensor,
        output_size: Optional[List[int]] = ...) -> Tensor: ...


class MaxUnpool2d(_MaxUnpoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t

    def __init__(
        self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = ...,
        padding: _size_2_t = ...) -> None: ...

    def forward(
        self, input: Tensor, indices: Tensor,
        output_size: Optional[List[int]] = ...) -> Tensor: ...


class MaxUnpool3d(_MaxUnpoolNd):
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t

    def __init__(
        self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = ...,
        padding: _size_3_t = ...) -> None: ...

    def forward(
        self, input: Tensor, indices: Tensor,
        output_size: Optional[List[int]] = ...) -> Tensor: ...


class _AvgPoolNd(Module):
    __constants__: Incomplete
    def extra_repr(self) -> str: ...


class AvgPool1d(_AvgPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(
        self, kernel_size: _size_1_t, stride: _size_1_t = ...,
        padding: _size_1_t = ..., ceil_mode: bool = ...,
        count_include_pad: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class AvgPool2d(_AvgPoolNd):
    __constants__: Incomplete
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: Incomplete

    def __init__(
        self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = ...,
        padding: _size_2_t = ..., ceil_mode: bool = ...,
        count_include_pad: bool = ...,
        divisor_override: Optional[int] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class AvgPool3d(_AvgPoolNd):
    __constants__: Incomplete
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: Incomplete

    def __init__(
        self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = ...,
        padding: _size_3_t = ..., ceil_mode: bool = ...,
        count_include_pad: bool = ...,
        divisor_override: Optional[int] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class FractionalMaxPool2d(Module):
    __constants__: Incomplete
    kernel_size: _size_2_t
    return_indices: bool
    output_size: _size_2_t
    output_ratio: _ratio_2_t

    def __init__(
        self, kernel_size: _size_2_t, output_size: Optional[_size_2_t] = ...,
        output_ratio: Optional[_ratio_2_t] = ..., return_indices: bool = ...,
        _random_samples: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor): ...


class FractionalMaxPool3d(Module):
    __constants__: Incomplete
    kernel_size: _size_3_t
    return_indices: bool
    output_size: _size_3_t
    output_ratio: _ratio_3_t

    def __init__(
        self, kernel_size: _size_3_t, output_size: Optional[_size_3_t] = ...,
        output_ratio: Optional[_ratio_3_t] = ..., return_indices: bool = ...,
        _random_samples: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor): ...


class _LPPoolNd(Module):
    __constants__: Incomplete
    norm_type: float
    ceil_mode: bool
    kernel_size: Incomplete
    stride: Incomplete

    def __init__(
        self, norm_type: float, kernel_size: _size_any_t,
        stride: Optional[
                _size_any_t] = ..., ceil_mode: bool = ...) -> None: ...

    def extra_repr(self) -> str: ...


class LPPool1d(_LPPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t
    def forward(self, input: Tensor) -> Tensor: ...


class LPPool2d(_LPPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    def forward(self, input: Tensor) -> Tensor: ...


class _AdaptiveMaxPoolNd(Module):
    __constants__: Incomplete
    return_indices: bool
    output_size: Incomplete

    def __init__(
        self, output_size: _size_any_opt_t,
        return_indices: bool = ...) -> None: ...

    def extra_repr(self) -> str: ...


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    output_size: _size_1_t
    def forward(self, input: Tensor) -> Tensor: ...


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    output_size: _size_2_opt_t
    def forward(self, input: Tensor): ...


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    output_size: _size_3_opt_t
    def forward(self, input: Tensor): ...


class _AdaptiveAvgPoolNd(Module):
    __constants__: Incomplete
    output_size: Incomplete
    def __init__(self, output_size: _size_any_opt_t) -> None: ...
    def extra_repr(self) -> str: ...


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    output_size: _size_1_t
    def forward(self, input: Tensor) -> Tensor: ...


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    output_size: _size_2_opt_t
    def forward(self, input: Tensor) -> Tensor: ...


class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):
    output_size: _size_3_opt_t
    def forward(self, input: Tensor) -> Tensor: ...
