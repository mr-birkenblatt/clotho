# Stubs for pandas.core.indexes.period (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-import,unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods,no-name-in-module,function-redefined
# pylint: disable=redefined-outer-name,too-many-ancestors,super-init-not-called

from pandas._libs.tslibs.period import Period as Period
from pandas.core.indexes.datetimelike import (
    DatetimeIndexOpsMixin,
    DatetimelikeDelegateMixin,
)
from pandas.core.indexes.datetimes import Int64Index
from typing import Any, Optional


class PeriodDelegateMixin(DatetimelikeDelegateMixin):
    ...


class PeriodIndex(  # type: ignore
        DatetimeIndexOpsMixin,
        Int64Index, PeriodDelegateMixin):
    def __new__(
            cls, data: Optional[Any] = ...,
            ordinal: Optional[Any] = ..., freq: Optional[Any] = ...,
            start: Optional[Any] = ..., end: Optional[Any] = ...,
            periods: Optional[Any] = ..., tz: Optional[Any] = ...,
            dtype: Optional[Any] = ..., copy: bool = ...,
            name: Optional[Any] = ..., **fields: Any) -> Any:
        ...

    @property
    def values(self) -> Any:
        ...

    @property
    def freq(self) -> Any:
        ...

    @freq.setter
    def freq(self, value: Any) -> None:
        ...

    def __contains__(self, key: Any) -> Any:
        ...

    def __array__(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def __array_wrap__(self, result: Any, context: Optional[Any] = ...) -> Any:
        ...

    def asof_locs(self, where: Any, mask: Any) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ..., how: str = ...) -> Any:
        ...

    def searchsorted(
            self, value: Any, side: str = ...,
            sorter: Optional[Any] = ...) -> Any:
        ...

    @property
    def is_all_dates(self) -> Any:
        ...

    @property
    def is_full(self) -> Any:
        ...

    @property
    def inferred_type(self) -> Any:
        ...

    def get_value(self, series: Any, key: Any) -> Any:
        ...

    def get_indexer(
            self, target: Any, method: Optional[Any] = ...,
            limit: Optional[Any] = ...,
            tolerance: Optional[Any] = ...) -> Any:
        ...

    def get_indexer_non_unique(self, target: Any) -> Any:
        ...

    def unique(self, level: Optional[Any] = ...) -> Any:
        ...

    def get_loc(
            self, key: Any, method: Optional[Any] = ...,
            tolerance: Optional[Any] = ...) -> Any:
        ...

    def insert(self, loc: Any, item: Any) -> Any:
        ...

    def join(
            self, other: Any, how: str = ..., level: Optional[Any] = ...,
            return_indexers: bool = ..., sort: bool = ...) -> Any:
        ...

    def intersection(self, other: Any, sort: bool = ...) -> Any:
        ...

    @property
    def flags(self) -> Any:
        ...

    def item(self) -> Any:
        ...

    @property
    def data(self) -> Any:
        ...

    @property
    def base(self) -> Any:
        ...

    def memory_usage(self, deep: bool = ...) -> Any:
        ...


def period_range(
        start: Optional[Any] = ..., end: Optional[Any] = ...,
        periods: Optional[Any] = ..., freq: Optional[Any] = ...,
        name: Optional[Any] = ...) -> Any:
    ...
