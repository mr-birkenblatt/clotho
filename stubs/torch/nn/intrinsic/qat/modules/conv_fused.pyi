# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import TypeVar

import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.qat as nnqat
from _typeshed import Incomplete
from torch.nn import init as init
from torch.nn.parameter import Parameter as Parameter
from torch.nn.utils import fuse_conv_bn_weights as fuse_conv_bn_weights


MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)


class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    qconfig: Incomplete
    freeze_bn: Incomplete
    bn: Incomplete
    weight_fake_quant: Incomplete
    bias: Incomplete

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
        dilation, transposed, output_padding, groups, bias, padding_mode,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ..., dim: int = ...) -> None: ...

    def reset_running_stats(self) -> None: ...
    def reset_bn_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def update_bn_stats(self): ...
    def freeze_bn_stats(self): ...
    def extra_repr(self): ...
    def forward(self, input): ...
    training: Incomplete
    def train(self, mode: bool = ...): ...
    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...


class ConvBn1d(_ConvBnNd, nn.Conv1d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: Incomplete | None = ..., padding_mode: str = ...,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...


class ConvBnReLU1d(ConvBn1d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: Incomplete | None = ..., padding_mode: str = ...,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class ConvBn2d(_ConvBnNd, nn.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: Incomplete | None = ..., padding_mode: str = ...,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...


class ConvBnReLU2d(ConvBn2d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: Incomplete | None = ..., padding_mode: str = ...,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class ConvBn3d(_ConvBnNd, nn.Conv3d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: Incomplete | None = ..., padding_mode: str = ...,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...


class ConvBnReLU3d(ConvBn3d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: Incomplete | None = ..., padding_mode: str = ...,
        eps: float = ..., momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


def update_bn_stats(mod) -> None: ...


def freeze_bn_stats(mod) -> None: ...
