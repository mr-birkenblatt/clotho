# Stubs for pandas.core.base (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-import,unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods,function-redefined
# pylint: disable=redefined-outer-name,too-many-ancestors,super-init-not-called
# pylint: disable=too-many-arguments,keyword-arg-before-vararg

from typing import Any, Optional
from pandas.core.accessor import DirNamesMixin
from pandas.core.arrays import ExtensionArray


class StringMixin:
    ...


class PandasObject(DirNamesMixin):
    def __sizeof__(self) -> Any:
        ...


class NoNewAttributesMixin:
    def __setattr__(self, key: Any, value: Any) -> None:
        ...


class GroupByError(Exception):
    ...


class DataError(GroupByError):
    ...


class SpecificationError(GroupByError):
    ...


class SelectionMixin:
    def ndim(self) -> Any:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> None:
        ...

    agg: Any = ...


class IndexOpsMixin:
    __array_priority__: int = ...

    def transpose(self, *args: Any, **kwargs: Any) -> Any:
        ...

    T: Any = ...

    @property
    def shape(self) -> Any:
        ...

    @property
    def ndim(self) -> Any:
        ...

    def item(self) -> Any:
        ...

    @property
    def data(self) -> Any:
        ...

    @property
    def itemsize(self) -> Any:
        ...

    @property
    def nbytes(self) -> Any:
        ...

    @property
    def strides(self) -> Any:
        ...

    @property
    def size(self) -> Any:
        ...

    @property
    def flags(self) -> Any:
        ...

    @property
    def base(self) -> Any:
        ...

    @property
    def array(self) -> ExtensionArray:
        ...

    def to_numpy(self, dtype: Optional[Any] = ..., copy: bool = ...) -> Any:
        ...

    @property
    def empty(self) -> Any:
        ...

    def max(
            self, axis: Optional[Any] = ..., skipna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def argmax(
            self, axis: Optional[Any] = ..., skipna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def min(
            self, axis: Optional[Any] = ..., skipna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def argmin(
            self, axis: Optional[Any] = ..., skipna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def tolist(self) -> Any:
        ...

    to_list: Any = ...

    def __iter__(self) -> Any:
        ...

    def hasnans(self) -> Any:
        ...

    def value_counts(
            self, normalize: bool = ..., sort: bool = ...,
            ascending: bool = ..., bins: Optional[Any] = ...,
            dropna: bool = ...) -> Any:
        ...

    def unique(self) -> Any:
        ...

    def nunique(self, dropna: bool = ...) -> Any:
        ...

    @property
    def is_unique(self) -> Any:
        ...

    @property
    def is_monotonic(self) -> Any:
        ...

    is_monotonic_increasing: Any = ...

    @property
    def is_monotonic_decreasing(self) -> Any:
        ...

    def memory_usage(self, deep: bool = ...) -> Any:
        ...

    def factorize(self, sort: bool = ..., na_sentinel: int = ...) -> Any:
        ...

    def searchsorted(
            self, value: Any, side: str = ...,
            sorter: Optional[Any] = ...) -> Any:
        ...

    def drop_duplicates(self, keep: str = ..., inplace: bool = ...) -> Any:
        ...

    def duplicated(self, keep: str = ...) -> Any:
        ...
