# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Tuple, TypeVar, Union

import torch.nn as nn
from _typeshed import Incomplete
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)


class _ConvNd(nn.modules.conv._ConvNd):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int,
        ...], stride: Tuple[int, ...], padding: Tuple[int, ...],
        dilation: Tuple[int, ...], transposed: bool,
        output_padding: Tuple[int, ...], groups: int, bias: bool,
        padding_mode: str, qconfig: Incomplete | None = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @staticmethod
    def from_float(cls, mod): ...
    def to_float(self): ...


class Conv1d(_ConvNd, nn.Conv1d):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: Union[str, _size_1_t] = ...,
        dilation: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., qconfig: Incomplete | None = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    @classmethod
    def from_float(cls, mod): ...


class Conv2d(_ConvNd, nn.Conv2d):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
        stride: _size_2_t = ..., padding: Union[str, _size_2_t] = ...,
        dilation: _size_2_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., qconfig: Incomplete | None = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class Conv3d(_ConvNd, nn.Conv3d):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_3_t,
        stride: _size_3_t = ..., padding: Union[str, _size_3_t] = ...,
        dilation: _size_3_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., qconfig: Incomplete | None = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
