# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.jit.annotations import BroadcastingList2 as BroadcastingList2


def avg_pool2d(
    input, kernel_size, stride: Incomplete | None = ..., padding: int = ...,
    ceil_mode: bool = ..., count_include_pad: bool = ...,
    divisor_override: Incomplete | None = ...): ...


def avg_pool3d(
    input, kernel_size, stride: Incomplete | None = ..., padding: int = ...,
    ceil_mode: bool = ..., count_include_pad: bool = ...,
    divisor_override: Incomplete | None = ...): ...


def adaptive_avg_pool2d(
    input: Tensor, output_size: BroadcastingList2[int]) -> Tensor: ...


def adaptive_avg_pool3d(
    input: Tensor, output_size: BroadcastingList2[int]) -> Tensor: ...


def conv1d(
    input, weight, bias, stride: int = ..., padding: int = ...,
    dilation: int = ..., groups: int = ..., padding_mode: str = ...,
    scale: float = ..., zero_point: int = ..., dtype=...): ...


def conv2d(
    input, weight, bias, stride: int = ..., padding: int = ...,
    dilation: int = ..., groups: int = ..., padding_mode: str = ...,
    scale: float = ..., zero_point: int = ..., dtype=...): ...


def conv3d(
    input, weight, bias, stride: int = ..., padding: int = ...,
    dilation: int = ..., groups: int = ..., padding_mode: str = ...,
    scale: float = ..., zero_point: int = ..., dtype=...): ...


def interpolate(
    input, size: Incomplete | None = ...,
    scale_factor: Incomplete | None = ..., mode: str = ...,
    align_corners: Incomplete | None = ...): ...


def linear(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = ...,
    scale: Optional[float] = ..., zero_point: Optional[
            int] = ...) -> Tensor: ...


def max_pool1d(
    input, kernel_size, stride: Incomplete | None = ..., padding: int = ...,
    dilation: int = ..., ceil_mode: bool = ...,
    return_indices: bool = ...): ...


def max_pool2d(
    input, kernel_size, stride: Incomplete | None = ..., padding: int = ...,
    dilation: int = ..., ceil_mode: bool = ...,
    return_indices: bool = ...): ...


def celu(
    input: Tensor, scale: float, zero_point: int,
    alpha: float = ...) -> Tensor: ...


def leaky_relu(
    input: Tensor, negative_slope: float = ..., inplace: bool = ...,
    scale: Optional[float] = ..., zero_point: Optional[int] = ...): ...


def hardtanh(
    input: Tensor, min_val: float = ..., max_val: float = ...,
    inplace: bool = ...) -> Tensor: ...


def hardswish(input: Tensor, scale: float, zero_point: int) -> Tensor: ...


def threshold(input: Tensor, threshold: float, value: float) -> Tensor: ...


def elu(
    input: Tensor, scale: float, zero_point: int,
    alpha: float = ...) -> Tensor: ...


def hardsigmoid(input: Tensor, inplace: bool = ...) -> Tensor: ...


def clamp(input: Tensor, min_: float, max_: float) -> Tensor: ...


def upsample(
    input, size: Incomplete | None = ...,
    scale_factor: Incomplete | None = ..., mode: str = ...,
    align_corners: Incomplete | None = ...): ...


def upsample_bilinear(
    input, size: Incomplete | None = ...,
    scale_factor: Incomplete | None = ...): ...


def upsample_nearest(
    input, size: Incomplete | None = ...,
    scale_factor: Incomplete | None = ...): ...
