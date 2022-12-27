# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.types import _dtype as DType


DimOrDims = Optional[Union[int, Tuple[int], List[int]]]


def sum(
    input: Tensor, dim: DimOrDims = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def prod(
    input: Tensor, dim: DimOrDims = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def cumsum(
    input: Tensor, dim: int, *, dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def cumprod(
    input: Tensor, dim: int, *, dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def amax(
    input: Tensor, dim: DimOrDims = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def amin(
    input: Tensor, dim: DimOrDims = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def argmax(
    input: Tensor, dim: int = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def argmin(
    input: Tensor, dim: int = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def mean(
    input: Tensor, dim: DimOrDims = ..., *, keepdim: Optional[bool] = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def median(
    input: Tensor, dim: int = ..., *, keepdim: bool = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def norm(
    input: Tensor, ord: Optional[float] = ..., dim: DimOrDims = ..., *,
        keepdim: Optional[bool] = ..., dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def var(
    input: Tensor, dim: DimOrDims = ..., unbiased: Optional[bool] = ..., *,
        keepdim: Optional[bool] = ..., dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def std(
    input: Tensor, dim: DimOrDims = ..., unbiased: Optional[bool] = ..., *,
        keepdim: Optional[bool] = ..., dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def softmax(
    input: Tensor, dim: int, *, dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def log_softmax(
    input: Tensor, dim: int, *, dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def softmin(
    input: Tensor, dim: int, *, dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...


def normalize(
    input: Tensor, ord: float, dim: int, *, eps: float = ...,
        dtype: Optional[DType] = ...,
        mask: Optional[Tensor] = ...) -> Tensor: ...
