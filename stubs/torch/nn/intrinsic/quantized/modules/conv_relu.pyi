# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch.nn.quantized as nnq
from _typeshed import Incomplete
from torch.nn.utils import fuse_conv_bn_weights as fuse_conv_bn_weights


class ConvReLU1d(nnq.Conv1d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...


class ConvReLU2d(nnq.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...


class ConvReLU3d(nnq.Conv3d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point): ...
