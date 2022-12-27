# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional, TypeVar

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch._ops import ops as ops
from torch.nn.common_types import _size_1_t
from torch.nn.quantized.modules.utils import (
    WeightedQuantizedModule as WeightedQuantizedModule,
)
from torch.nn.utils import fuse_conv_bn_weights as fuse_conv_bn_weights


class _ConvNd(WeightedQuantizedModule):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(self, qweight, bias_float) -> None: ...
    def bias(self) -> None: ...
    def extra_repr(self): ...
    def __deepcopy__(self, memo): ...
    def __copy__(self): ...

    @classmethod
    def get_qconv(
        cls, mod, activation_post_process,
        weight_post_process: Incomplete | None = ...): ...

    @staticmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...


class Conv1d(_ConvNd):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: _size_1_t = ...,
        dilation: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class Conv2d(_ConvNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class Conv3d(_ConvNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)


class _ConvTransposeNd(_ConvNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
        dilation, transposed, output_padding, groups, bias, padding_mode,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    @classmethod
    def from_float(cls, mod): ...
    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...


class ConvTranspose1d(_ConvTransposeNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., output_padding: int = ..., groups: int = ...,
        bias: bool = ..., dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...


class ConvTranspose2d(_ConvTransposeNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., output_padding: int = ..., groups: int = ...,
        bias: bool = ..., dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...


class ConvTranspose3d(_ConvTransposeNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., output_padding: int = ..., groups: int = ...,
        bias: bool = ..., dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    def weight(self): ...
    def bias(self): ...
    def forward(self, input): ...
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point): ...
