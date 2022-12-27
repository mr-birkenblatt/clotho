# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from _typeshed import Incomplete
from torch._prims.utils import (
    DimsSequenceType,
    DimsType,
    NumberType,
    ShapeType,
    StrideType,
    TensorLikeType,
    TensorSequenceType,
)


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME: Incomplete
    SAME_OR_REAL: Incomplete
    OP_MATH: Incomplete
    ALWAYS_BOOL: Incomplete

abs: Incomplete
acos: Incomplete
acosh: Incomplete
asin: Incomplete
atan: Incomplete
bitwise_not: Incomplete
ceil: Incomplete
cos: Incomplete
cosh: Incomplete
digamma: Incomplete
erf: Incomplete
erfinv: Incomplete
erfc: Incomplete
exp: Incomplete
expm1: Incomplete
floor: Incomplete
isfinite: Incomplete
isinf: Incomplete
isnan: Incomplete
lgamma: Incomplete
log: Incomplete
log1p: Incomplete
log2: Incomplete
neg: Incomplete
reciprocal: Incomplete
round: Incomplete
sign: Incomplete
sin: Incomplete
sinh: Incomplete
sqrt: Incomplete
square: Incomplete
tan: Incomplete
tanh: Incomplete


def add(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType,
            NumberType], *, alpha: Optional[NumberType] = ...): ...


atan2: Incomplete
bitwise_and: Incomplete
bitwise_left_shift: Incomplete
bitwise_or: Incomplete
bitwise_right_shift: Incomplete
bitwise_xor: Incomplete
eq: Incomplete


def float_power(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType,
            NumberType]) -> Tensor: ...


ge: Incomplete
gt: Incomplete
igamma: Incomplete
igammac: Incomplete


def isclose(
    a: TensorLikeType, b: TensorLikeType, rtol: float = ...,
    atol: float = ..., equal_nan: bool = ...) -> TensorLikeType: ...


le: Incomplete
logical_and: Incomplete
logical_or: Incomplete
lt: Incomplete
maximum: Incomplete
minimum: Incomplete
mul: Incomplete
ne: Incomplete
nextafter: Incomplete
pow: Incomplete


def sub(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType,
            NumberType], *, alpha: Optional[NumberType] = ...): ...


true_divide: Incomplete


def where(
    pred: Tensor, a: Optional[Union[TensorLikeType, NumberType]] = ...,
    b: Optional[Union[TensorLikeType, NumberType]] = ...): ...


def clone(
    a: TensorLikeType, *,
    memory_format: torch.memory_format = ...) -> TensorLikeType: ...


def copy_to(a: Tensor, b: Tensor, *, allow_cross_device: bool = ...): ...


def sum(
    a: Tensor, dim: Union[Optional[int], Optional[List[int]]] = ...,
    keepdim: bool = ..., *, dtype: Incomplete | None = ...,
    out: Optional[Tensor] = ...): ...


def amin(
    a: Tensor, dim: Union[Optional[int], Optional[List[int]]] = ...,
    keepdim: bool = ..., *, out: Optional[Tensor] = ...): ...


def amax(
    a: Tensor, dim: Union[Optional[int], Optional[List[int]]] = ...,
    keepdim: bool = ..., *, out: Optional[Tensor] = ...): ...


def as_strided(
    a: TensorLikeType, size: ShapeType, stride: StrideType,
    storage_offset: int = ...) -> TensorLikeType: ...


def cat(tensors: TensorSequenceType, dim: int = ...) -> TensorLikeType: ...


def chunk(
    a: TensorLikeType, chunks: int, dim: int = ...) -> Tuple[
        TensorLikeType, ...]: ...


def flatten(
    a: TensorLikeType, start_dim: int = ...,
    end_dim: int = ...) -> TensorLikeType: ...


def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType: ...


def narrow(
    a: TensorLikeType, dim: int, start: int,
    length: int) -> TensorLikeType: ...


def permute(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType: ...


def reshape(a: TensorLikeType, shape: ShapeType) -> TensorLikeType: ...


def stack(tensors: TensorSequenceType, dim: int = ...) -> TensorLikeType: ...


def squeeze(a: TensorLikeType, dim: Optional[int] = ...) -> TensorLikeType: ...


def tensor_split(
    a: TensorLikeType, indices_or_sections: Union[Tensor, DimsType],
    dim: int = ...) -> Tuple[TensorLikeType, ...]: ...


def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType: ...


swap_axes = transpose


def unsqueeze(a: TensorLikeType, dim: int) -> TensorLikeType: ...


def view(a: TensorLikeType, shape: ShapeType) -> TensorLikeType: ...


def empty(
    *shape, dtype: Optional[torch.dtype] = ...,
    device: Optional[
            torch.device] = ...,
    requires_grad: bool = ...) -> TensorLikeType: ...


def empty_like(
    a: TensorLikeType, *, dtype: Optional[torch.dtype] = ...,
    device: Optional[
            torch.device] = ...,
    requires_grad: bool = ...) -> TensorLikeType: ...


def full(
    shape: ShapeType, fill_value: NumberType, *, dtype: torch.dtype,
    device: torch.device, requires_grad: bool) -> TensorLikeType: ...


def full_like(
    a: TensorLikeType, fill_value: NumberType, *,
    dtype: Optional[torch.dtype] = ..., device: Optional[torch.device] = ...,
    requires_grad: bool = ...) -> TensorLikeType: ...


def ones_like(
    a: TensorLikeType, *, dtype: Optional[torch.dtype] = ...,
    device: Optional[
            torch.device] = ...,
    requires_grad: bool = ...) -> TensorLikeType: ...


def zeros_like(
    a: TensorLikeType, *, dtype: Optional[torch.dtype] = ...,
    device: Optional[
            torch.device] = ...,
    requires_grad: bool = ...) -> TensorLikeType: ...
