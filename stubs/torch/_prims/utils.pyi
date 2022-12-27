# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from _typeshed import Incomplete
from torch._C._nvfuser import DataType as DataType


def getnvFuserDtype(dtype: torch.dtype): ...


ShapeType: Incomplete
StrideType = Union[List[int], Tuple[int, ...]]
DimsType = Union[int, List[int], Tuple[int, ...]]
DimsSequenceType = Union[List[int], Tuple[int, ...]]
NumberType = Union[bool, int, float, complex]
Number: Incomplete


class TensorMeta(torch.Tensor):
    node: Optional[Any]
    tname: str

    @staticmethod
    def __new__(
        cls, tensorlike: Optional[Union[TensorMeta, NumberType,
        torch.Tensor]] = ..., *, shape: Optional[ShapeType] = ...,
        strides: Optional[StrideType] = ...,
        dtype: Optional[torch.dtype] = ...,
        device: Optional[Union[torch.device, str]] = ...): ...

    @classmethod
    def __torch_function__(
        cls, func: Callable, types: Sequence, args: Sequence[Any] = ...,
        kwargs: Optional[Dict] = ...): ...

    @classmethod
    def __torch_dispatch__(
        cls, func, types, args=...,
        kwargs: Incomplete | None = ...) -> None: ...

    def __format__(self, format_spec): ...


TensorLikeType: Incomplete
TensorLike: Incomplete
TensorSequenceType = Union[List[TensorLikeType], Tuple[TensorLikeType, ...]]


def compare_tensor_meta(a: TensorLikeType, b: TensorLikeType): ...


def check_significant_strides(
    a: TensorLikeType, b: TensorLikeType) -> Tuple[bool, Optional[int]]: ...


def is_contiguous(a: TensorLikeType) -> bool: ...


def compute_elementwise_output_strides(*tensors) -> Tuple[int, ...]: ...


def validate_dim_length(length: int): ...


def validate_shape(shape: ShapeType): ...


def validate_strides(strides: StrideType): ...


def validate_idx(rank: int, idx: int): ...


def validate_dimension_indices(rank: int, indices: DimsSequenceType): ...


def validate_exclusive_idx(rank: int, ex_idx: int): ...


def canonicalize_dim(rank: int, idx: int) -> int: ...


def canonicalize_dims(rank: int, indices: DimsType) -> DimsType: ...


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool: ...


def is_same_shape(a: Sequence, b: Sequence) -> bool: ...


def is_cpu_scalar_tensor(a: Any) -> bool: ...


def check_same_device(*args, allow_cpu_scalar_tensors) -> None: ...


def check_same_shape(*args, allow_cpu_scalar_tensors) -> None: ...


def is_boolean_dtype(dtype: torch.dtype) -> bool: ...


def is_integer_dtype(dtype: torch.dtype) -> bool: ...


def is_float_dtype(dtype: torch.dtype) -> bool: ...


def is_complex_dtype(dtype: torch.dtype) -> bool: ...


def corresponding_real_dtype(dtype: torch.dtype) -> torch.dtype: ...


def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype: ...


def dtype_to_type(dtype: torch.dtype) -> type: ...


def type_to_dtype(typ: type) -> torch.dtype: ...


def get_higher_type(a: type, b: type) -> type: ...


def get_higher_dtype(
    a: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
    b: Optional[Union[torch.dtype, TensorLikeType,
    NumberType]]) -> Optional[torch.dtype]: ...


def is_weakly_lesser_type(a: type, b: type) -> bool: ...


def can_safe_cast_to(
    *, cast_to: torch.dtype, cast_from: torch.dtype) -> bool: ...


def check_same_dtype(*args) -> None: ...


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT: Incomplete
    NO_OPMATH: Incomplete
    INT_TO_FLOAT: Incomplete
    ALWAYS_BOOL: Incomplete
    COMPLEX_TO_FLOAT: Incomplete
    BOOL_TO_LONG: Incomplete


def elementwise_dtypes(
    *_args,
    type_promotion_kind:
    ELEMENTWISE_TYPE_PROMOTION_KIND) -> Tuple[torch.dtype, torch.dtype]: ...


def wrap_device(d: Union[str, torch.device]) -> torch.device: ...


def make_contiguous_strides_for(shape: ShapeType) -> Tuple[int, ...]: ...


def compute_reduction_output_shape(
    shape: ShapeType, dimensions: Sequence) -> Tuple[int, ...]: ...


def validate_no_repeating_dims(dims: Sequence): ...


def reduction_dims(
    shape: ShapeType, dims: Optional[Sequence]) -> Tuple[int, ...]: ...


def check_in_bounds_for_storage(
    a: torch._TypedStorage, shape: ShapeType, strides: StrideType,
    storage_offset: int): ...
