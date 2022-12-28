# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch.nn.quantized.modules as nnq
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._ops import ops as ops
from torch.nn.common_types import _size_1_t


class Conv1d(nnq.Conv1d):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: _size_1_t = ...,
        dilation: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ..., reduce_range: bool = ...) -> None: ...

    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...


class Conv2d(nnq.Conv2d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...


class Conv3d(nnq.Conv3d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., dilation: int = ..., groups: int = ...,
        bias: bool = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...


class ConvTranspose1d(nnq.ConvTranspose1d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., output_padding: int = ..., groups: int = ...,
        bias: bool = ..., dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...


class ConvTranspose2d(nnq.ConvTranspose2d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., output_padding: int = ..., groups: int = ...,
        bias: bool = ..., dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...


class ConvTranspose3d(nnq.ConvTranspose3d):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride: int = ...,
        padding: int = ..., output_padding: int = ..., groups: int = ...,
        bias: bool = ..., dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, reduce_range: bool = ...) -> Tensor: ...
