# Stubs for pandas.core.indexes.interval (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-import,unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods,no-self-use,function-redefined
# pylint: disable=redefined-outer-name,too-many-ancestors,super-init-not-called
# pylint: disable=too-many-arguments

import numpy as np
from pandas._libs.interval import IntervalMixin, Interval as Interval
from pandas._typing import AnyArrayLike
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.indexes.base import Index
from typing import Any, Optional, Tuple, Union


class SetopCheck:
    op_name: Any = ...

    def __init__(self, op_name: Any) -> None:
        ...

    def __call__(self, setop: Any) -> Any:
        ...


class IntervalIndex(IntervalMixin, Index):
    def __new__(
            cls, data: Any, closed: Optional[Any] = ...,
            dtype: Optional[Any] = ..., copy: bool = ...,
            name: Optional[Any] = ...,
            verify_integrity: bool = ...) -> Any:
        ...

    @classmethod
    def from_breaks(
            cls, breaks: Any, closed: str = ...,
            name: Optional[Any] = ..., copy: bool = ...,
            dtype: Optional[Any] = ...) -> Any:
        ...

    @classmethod
    def from_arrays(
            cls, left: Any, right: Any, closed: str = ...,
            name: Optional[Any] = ..., copy: bool = ...,
            dtype: Optional[Any] = ...) -> Any:
        ...

    @classmethod
    def from_intervals(
            cls, data: Any, closed: Optional[Any] = ...,
            name: Optional[Any] = ..., copy: bool = ...,
            dtype: Optional[Any] = ...) -> Any:
        ...

    @classmethod
    def from_tuples(
            cls, data: Any, closed: str = ...,
            name: Optional[Any] = ..., copy: bool = ...,
            dtype: Optional[Any] = ...) -> Any:
        ...

    def __contains__(self, key: Any) -> Any:
        ...

    def to_tuples(self, na_tuple: bool = ...) -> Any:
        ...

    @property
    def left(self) -> Any:
        ...

    @property
    def right(self) -> Any:
        ...

    @property
    def closed(self) -> Any:
        ...

    def set_closed(self, closed: Any) -> Any:
        ...

    @property
    def length(self) -> Any:
        ...

    @property
    def size(self) -> Any:
        ...

    @property
    def itemsize(self) -> Any:
        ...

    def __len__(self) -> Any:
        ...

    def values(self) -> Any:
        ...

    def __array__(self, result: Optional[Any] = ...) -> Any:
        ...

    def __array_wrap__(self, result: Any, context: Optional[Any] = ...) -> Any:
        ...

    def __reduce__(self) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Any:
        ...

    def dtype(self) -> Any:
        ...

    @property
    def inferred_type(self) -> Any:
        ...

    def memory_usage(self, deep: bool = ...) -> Any:
        ...

    def mid(self) -> Any:
        ...

    def is_monotonic(self) -> Any:
        ...

    def is_monotonic_increasing(self) -> Any:
        ...

    def is_monotonic_decreasing(self) -> Any:
        ...

    def is_unique(self) -> Any:
        ...

    def is_non_overlapping_monotonic(self) -> Any:
        ...

    @property
    def is_overlapping(self) -> Any:
        ...

    def get_indexer(
            self, target: AnyArrayLike, method: Optional[str] = ...,
            limit: Optional[int] = ...,
            tolerance: Optional[Any] = ...) -> np.ndarray:
        ...

    def get_indexer_non_unique(
            self, target: AnyArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def get_indexer_for(
            self, target: AnyArrayLike, **kwargs: Any) -> np.ndarray:
        ...

    def get_value(self, series: ABCSeries, key: Any) -> Any:
        ...

    def where(self, cond: Any, other: Optional[Any] = ...) -> Any:
        ...

    def delete(self, loc: Any) -> Any:
        ...

    def insert(self, loc: Any, item: Any) -> Any:
        ...

    def take(
            self, indices: Any, axis: int = ..., allow_fill: bool = ...,
            fill_value: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def __getitem__(self, value: Any) -> Any:
        ...

    def argsort(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def equals(self, other: Any) -> Any:
        ...

    def contains(self, other: Any) -> Any:
        ...

    def overlaps(self, other: Any) -> Any:
        ...

    def intersection(
            self, other: IntervalIndex, sort: bool = ...) -> IntervalIndex:
        ...

    @property
    def is_all_dates(self) -> Any:
        ...

    union: Any = ...
    difference: Any = ...
    symmetric_difference: Any = ...


def interval_range(
        start: Optional[Any] = ..., end: Optional[Any] = ...,
        periods: Optional[Any] = ..., freq: Optional[Any] = ...,
        name: Optional[Any] = ..., closed: str = ...) -> Any:
    ...
