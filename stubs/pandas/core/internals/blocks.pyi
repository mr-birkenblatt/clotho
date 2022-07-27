# Stubs for pandas.core.internals.blocks (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,no-self-use,invalid-name
# pylint: disable=relative-beyond-top-level,line-too-long,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member,too-many-ancestors

from typing import Any, List, Optional
import pandas._libs.internals as libinternals
from pandas.core.base import PandasObject

class Block(PandasObject):
    is_numeric: bool = ...
    is_float: bool = ...
    is_integer: bool = ...
    is_complex: bool = ...
    is_datetime: bool = ...
    is_datetimetz: bool = ...
    is_timedelta: bool = ...
    is_bool: bool = ...
    is_object: bool = ...
    is_categorical: bool = ...
    is_extension: bool = ...
    ndim: Any = ...
    values: Any = ...

    def __init__(self, values: Any, placement: Any,
                 ndim: Optional[Any] = ...) -> None:
        ...

    @property
    def is_view(self) -> Any:
        ...

    @property
    def is_datelike(self) -> Any:
        ...

    def is_categorical_astype(self, dtype: Any) -> Any:
        ...

    def external_values(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def internal_values(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def formatting_values(self) -> Any:
        ...

    def get_values(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def get_block_values(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def to_dense(self) -> Any:
        ...

    @property
    def fill_value(self) -> Any:
        ...

    @property
    def mgr_locs(self) -> Any:
        ...

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: Any) -> None:
        ...

    @property
    def array_dtype(self) -> Any:
        ...

    def make_block(self, values: Any, placement: Optional[Any] = ...) -> Any:
        ...

    def make_block_same_class(self, values: Any, placement: Optional[Any] = ...,
                              ndim: Optional[Any] = ...,
                              dtype: Optional[Any] = ...) -> Any:
        ...

    def __len__(self) -> Any:
        ...

    def getitem_block(self, slicer: Any,
                      new_mgr_locs: Optional[Any] = ...) -> Any:
        ...

    @property
    def shape(self) -> Any:
        ...

    @property
    def dtype(self) -> Any:
        ...

    @property
    def ftype(self) -> Any:
        ...

    def merge(self, other: Any) -> Any:
        ...

    def concat_same_type(self, to_concat: Any,
                         placement: Optional[Any] = ...) -> Any:
        ...

    def iget(self, i: Any) -> Any:
        ...

    def set(self, locs: Any, values: Any) -> None:
        ...

    def delete(self, loc: Any) -> None:
        ...

    def apply(self, func: Any, **kwargs: Any) -> Any:
        ...

    def fillna(self, value: Any, limit: Optional[Any] = ...,
               inplace: bool = ..., downcast: Optional[Any] = ...) -> Any:
        ...

    def split_and_operate(self, mask: Any, f: Any, inplace: Any) -> Any:
        ...

    def downcast(self, dtypes: Optional[Any] = ...) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ..., errors: str = ...,
               values: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def convert(self, copy: bool = ..., **kwargs: Any) -> Any:
        ...

    def to_native_types(self, slicer: Optional[Any] = ..., na_rep: str = ...,
                        quoting: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def copy(self, deep: bool = ...) -> Any:
        ...

    def replace(self, to_replace: Any, value: Any, inplace: bool = ...,
                filter: Optional[Any] = ..., regex: bool = ...,
                convert: bool = ...) -> Any:
        ...

    def setitem(self, indexer: Any, value: Any) -> Any:
        ...

    def putmask(self, mask: Any, new: Any, align: bool = ...,
                inplace: bool = ..., axis: int = ...,
                transpose: bool = ...) -> Any:
        ...

    def coerce_to_target_dtype(self, other: Any) -> Any:
        ...

    def interpolate(self, method: str = ..., axis: int = ...,
                    index: Optional[Any] = ..., values: Optional[Any] = ...,
                    inplace: bool = ..., limit: Optional[Any] = ...,
                    limit_direction: str = ...,
                    limit_area: Optional[Any] = ...,
                    fill_value: Optional[Any] = ..., coerce: bool = ...,
                    downcast: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def take_nd(self, indexer: Any, axis: Any,
                new_mgr_locs: Optional[Any] = ...,
                fill_tuple: Optional[Any] = ...) -> Any:
        ...

    def diff(self, n: Any, axis: int = ...) -> Any:
        ...

    def shift(self, periods: Any, axis: int = ...,
              fill_value: Optional[Any] = ...) -> Any:
        ...

    def where(self, other: Any, cond: Any, align: bool = ...,
              errors: str = ..., try_cast: bool = ..., axis: int = ...) -> Any:
        ...

    def equals(self, other: Any) -> Any:
        ...

    def quantile(self, qs: Any, interpolation: str = ...,
                 axis: int = ...) -> Any:
        ...


class NonConsolidatableMixIn:
    def __init__(self, values: Any, placement: Any,
                 ndim: Optional[Any] = ...) -> None:
        ...

    @property
    def shape(self) -> Any:
        ...

    def iget(self, col: Any) -> Any:
        ...

    def should_store(self, value: Any) -> Any:
        ...

    values: Any = ...

    def set(self, locs: Any, values: Any, check: bool = ...) -> None:
        ...

    def putmask(self, mask: Any, new: Any, align: bool = ...,
                inplace: bool = ..., axis: int = ...,
                transpose: bool = ...) -> Any:
        ...


class ExtensionBlock(NonConsolidatableMixIn, Block):
    is_extension: bool = ...

    def __init__(self, values: Any, placement: Any,
                 ndim: Optional[Any] = ...) -> None:
        ...

    @property
    def fill_value(self) -> Any:
        ...

    @property
    def is_view(self) -> Any:
        ...

    @property
    def is_numeric(self) -> Any:  # type: ignore
        ...

    def setitem(self, indexer: Any, value: Any) -> Any:
        ...

    def get_values(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def to_dense(self) -> Any:
        ...

    def take_nd(self, indexer: Any, axis: int = ...,
                new_mgr_locs: Optional[Any] = ...,
                fill_tuple: Optional[Any] = ...) -> Any:
        ...

    def formatting_values(self) -> Any:
        ...

    def concat_same_type(self, to_concat: Any,
                         placement: Optional[Any] = ...) -> Any:
        ...

    def fillna(self, value: Any, limit: Optional[Any] = ...,
               inplace: bool = ..., downcast: Optional[Any] = ...) -> Any:
        ...

    def interpolate(self, method: str = ..., axis: int = ..., inplace: bool = ..., limit: Optional[Any] = ..., fill_value: Optional[Any] = ..., **kwargs: Any) -> Any:  # type: ignore
        ...

    def shift(self, periods: int, axis: libinternals.BlockPlacement = ..., fill_value: Any = ...) -> List[ExtensionBlock]:  # type: ignore
        ...

    def where(self, other: Any, cond: Any, align: bool = ...,
              errors: str = ..., try_cast: bool = ..., axis: int = ...) -> Any:
        ...


class ObjectValuesExtensionBlock(ExtensionBlock):
    def external_values(self, dtype: Optional[Any] = ...) -> Any:
        ...


class NumericBlock(Block):
    is_numeric: bool = ...


class FloatOrComplexBlock(NumericBlock):
    def equals(self, other: Any) -> Any:
        ...


class FloatBlock(FloatOrComplexBlock):
    is_float: bool = ...

    def to_native_types(self, slicer: Optional[Any] = ..., na_rep: str = ..., float_format: Optional[Any] = ..., decimal: str = ..., quoting: Optional[Any] = ..., **kwargs: Any) -> Any:  # type: ignore
        ...

    def should_store(self, value: Any) -> Any:
        ...


class ComplexBlock(FloatOrComplexBlock):
    is_complex: bool = ...

    def should_store(self, value: Any) -> Any:
        ...


class IntBlock(NumericBlock):
    is_integer: bool = ...

    def should_store(self, value: Any) -> Any:
        ...


class DatetimeLikeBlockMixin:
    @property
    def fill_value(self) -> Any:
        ...

    def get_values(self, dtype: Optional[Any] = ...) -> Any:
        ...


class DatetimeBlock(DatetimeLikeBlockMixin, Block):
    is_datetime: bool = ...

    def __init__(self, values: Any, placement: Any,
                 ndim: Optional[Any] = ...) -> None:
        ...

    def to_native_types(self, slicer: Optional[Any] = ..., na_rep: Optional[Any] = ..., date_format: Optional[Any] = ..., quoting: Optional[Any] = ..., **kwargs: Any) -> Any:  # type: ignore
        ...

    def should_store(self, value: Any) -> Any:
        ...

    def set(self, locs: Any, values: Any) -> None:
        ...

    def external_values(self) -> Any:  # type: ignore
        ...


class DatetimeTZBlock(ExtensionBlock, DatetimeBlock):
    is_datetimetz: bool = ...
    is_extension: bool = ...

    @property
    def is_view(self) -> Any:
        ...

    def get_values(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def to_dense(self) -> Any:
        ...

    def diff(self, n: Any, axis: int = ...) -> Any:
        ...

    def concat_same_type(self, to_concat: Any,
                         placement: Optional[Any] = ...) -> Any:
        ...

    def fillna(self, value: Any, limit: Optional[Any] = ...,
               inplace: bool = ..., downcast: Optional[Any] = ...) -> Any:
        ...

    def setitem(self, indexer: Any, value: Any) -> Any:
        ...

    def equals(self, other: Any) -> Any:
        ...


class TimeDeltaBlock(DatetimeLikeBlockMixin, IntBlock):
    is_timedelta: bool = ...
    is_numeric: bool = ...

    def __init__(self, values: Any, placement: Any,
                 ndim: Optional[Any] = ...) -> None:
        ...

    def fillna(self, value: Any, **kwargs: Any) -> Any:  # type: ignore
        ...

    def should_store(self, value: Any) -> Any:
        ...

    def to_native_types(self, slicer: Optional[Any] = ...,
                        na_rep: Optional[Any] = ...,
                        quoting: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def external_values(self, dtype: Optional[Any] = ...) -> Any:
        ...


class BoolBlock(NumericBlock):
    is_bool: bool = ...

    def should_store(self, value: Any) -> Any:
        ...

    def replace(self, to_replace: Any, value: Any, inplace: bool = ...,
                filter: Optional[Any] = ..., regex: bool = ...,
                convert: bool = ...) -> Any:
        ...


class ObjectBlock(Block):
    is_object: bool = ...

    def __init__(self, values: Any, placement: Optional[Any] = ...,
                 ndim: int = ...) -> None:
        ...

    @property
    def is_bool(self) -> Any:  # type: ignore
        ...

    def convert(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def should_store(self, value: Any) -> Any:
        ...

    def replace(self, to_replace: Any, value: Any, inplace: bool = ...,
                filter: Optional[Any] = ..., regex: bool = ...,
                convert: bool = ...) -> Any:
        ...


class CategoricalBlock(ExtensionBlock):
    is_categorical: bool = ...

    def __init__(self, values: Any, placement: Any,
                 ndim: Optional[Any] = ...) -> None:
        ...

    @property
    def array_dtype(self) -> Any:
        ...

    def to_dense(self) -> Any:
        ...

    def to_native_types(self, slicer: Optional[Any] = ..., na_rep: str = ...,
                        quoting: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def concat_same_type(self, to_concat: Any,
                         placement: Optional[Any] = ...) -> Any:
        ...

    def where(self, other: Any, cond: Any, align: bool = ...,
              errors: str = ..., try_cast: bool = ..., axis: int = ...) -> Any:
        ...


def get_block_type(values: Any, dtype: Optional[Any] = ...) -> Any:
    ...


def make_block(values: Any, placement: Any, klass: Optional[Any] = ...,
               ndim: Optional[Any] = ..., dtype: Optional[Any] = ...,
               fastpath: Optional[Any] = ...) -> Any:
    ...


def _extend_blocks(result: Any, blocks: Optional[Any] = None) -> Any:
    ...


def _block_shape(values: Any, ndim: int = 1,
                 shape: Optional[Any] = None) -> Any:
    ...


def _merge_blocks(blocks: Any, dtype: Optional[Any] = None,
                  _can_consolidate: bool = True) -> Any:
    ...


def _safe_reshape(arr: Any, new_shape: Any) -> Any:
    ...
