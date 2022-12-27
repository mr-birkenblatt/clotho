# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional, Tuple, Union

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._torch_docs import reproducibility_notes as reproducibility_notes
from torch.nn.parameter import Parameter as Parameter
from torch.nn.parameter import UninitializedParameter as UninitializedParameter

from .. import init as init
from ..common_types import _size_1_t, _size_2_t, _size_3_t
from .lazy import LazyModuleMixin as LazyModuleMixin
from .module import Module as Module


convolution_notes: Incomplete


class _ConvNd(Module):
    __constants__: Incomplete
    __annotations__: Incomplete
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    in_channels: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int,
            ...], stride: Tuple[int, ...], padding: Tuple[int, ...],
        dilation: Tuple[int, ...], transposed: bool,
        output_padding: Tuple[int, ...], groups: int, bias: bool,
        padding_mode: str, device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    def extra_repr(self): ...


class Conv1d(_ConvNd):
    __doc__: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: Union[str, _size_1_t] = ...,
        dilation: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class Conv2d(_ConvNd):
    __doc__: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
        stride: _size_2_t = ..., padding: Union[str, _size_2_t] = ...,
        dilation: _size_2_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class Conv3d(_ConvNd):
    __doc__: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_3_t,
        stride: _size_3_t = ..., padding: Union[str, _size_3_t] = ...,
        dilation: _size_3_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class _ConvTransposeNd(_ConvNd):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
        dilation, transposed, output_padding, groups, bias, padding_mode,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class ConvTranspose1d(_ConvTransposeNd):
    __doc__: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: _size_1_t = ...,
        output_padding: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        dilation: _size_1_t = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, input: Tensor,
        output_size: Optional[List[int]] = ...) -> Tensor: ...


class ConvTranspose2d(_ConvTransposeNd):
    __doc__: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
        stride: _size_2_t = ..., padding: _size_2_t = ...,
        output_padding: _size_2_t = ..., groups: int = ..., bias: bool = ...,
        dilation: _size_2_t = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, input: Tensor,
        output_size: Optional[List[int]] = ...) -> Tensor: ...


class ConvTranspose3d(_ConvTransposeNd):
    __doc__: Incomplete

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: _size_3_t,
        stride: _size_3_t = ..., padding: _size_3_t = ...,
        output_padding: _size_3_t = ..., groups: int = ..., bias: bool = ...,
        dilation: _size_3_t = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, input: Tensor,
        output_size: Optional[List[int]] = ...) -> Tensor: ...


class _ConvTransposeMixin(_ConvTransposeNd):
    def __init__(self, *args, **kwargs) -> None: ...


class _LazyConvXdMixin(LazyModuleMixin):
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input) -> None: ...


class LazyConv1d(_LazyConvXdMixin, Conv1d):
    cls_to_become: Incomplete
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete

    def __init__(
        self, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: _size_1_t = ...,
        dilation: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class LazyConv2d(_LazyConvXdMixin, Conv2d):
    cls_to_become: Incomplete
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete

    def __init__(
        self, out_channels: int, kernel_size: _size_2_t,
        stride: _size_2_t = ..., padding: _size_2_t = ...,
        dilation: _size_2_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class LazyConv3d(_LazyConvXdMixin, Conv3d):
    cls_to_become: Incomplete
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete

    def __init__(
        self, out_channels: int, kernel_size: _size_3_t,
        stride: _size_3_t = ..., padding: _size_3_t = ...,
        dilation: _size_3_t = ..., groups: int = ..., bias: bool = ...,
        padding_mode: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):
    cls_to_become: Incomplete
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete

    def __init__(
        self, out_channels: int, kernel_size: _size_1_t,
        stride: _size_1_t = ..., padding: _size_1_t = ...,
        output_padding: _size_1_t = ..., groups: int = ..., bias: bool = ...,
        dilation: _size_1_t = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class LazyConvTranspose2d(_LazyConvXdMixin, ConvTranspose2d):
    cls_to_become: Incomplete
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete

    def __init__(
        self, out_channels: int, kernel_size: _size_2_t,
        stride: _size_2_t = ..., padding: _size_2_t = ...,
        output_padding: _size_2_t = ..., groups: int = ..., bias: bool = ...,
        dilation: int = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class LazyConvTranspose3d(_LazyConvXdMixin, ConvTranspose3d):
    cls_to_become: Incomplete
    weight: Incomplete
    out_channels: Incomplete
    bias: Incomplete

    def __init__(
        self, out_channels: int, kernel_size: _size_3_t,
        stride: _size_3_t = ..., padding: _size_3_t = ...,
        output_padding: _size_3_t = ..., groups: int = ..., bias: bool = ...,
        dilation: _size_3_t = ..., padding_mode: str = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...
