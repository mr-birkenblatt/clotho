# Stubs for pandas.core.indexes.timedeltas (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-import,unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods,no-name-in-module,function-redefined
# pylint: disable=redefined-outer-name,too-many-ancestors,super-init-not-called

from pandas._libs.tslibs import Timedelta as Timedelta

from pandas.core.arrays import datetimelike as dtl
from pandas.core.indexes.datetimelike import (
    DatetimeIndexOpsMixin,
    DatetimelikeDelegateMixin,
)
from pandas.core.indexes.numeric import Int64Index
from typing import Any, Optional


class TimedeltaDelegateMixin(DatetimelikeDelegateMixin):
    ...


class TimedeltaIndex(  # type: ignore
        DatetimeIndexOpsMixin,
        dtl.TimelikeOps,
        Int64Index,
        TimedeltaDelegateMixin):
    def __new__(
            cls, data: Optional[Any] = ..., unit: Optional[Any] = ...,
            freq: Optional[Any] = ..., start: Optional[Any] = ...,
            end: Optional[Any] = ..., periods: Optional[Any] = ...,
            closed: Optional[Any] = ..., dtype: Any = ...,
            copy: bool = ..., name: Optional[Any] = ...,
            verify_integrity: Optional[Any] = ...) -> Any:
        ...

    __mul__: Any = ...
    __rmul__: Any = ...
    __floordiv__: Any = ...
    __rfloordiv__: Any = ...
    __mod__: Any = ...
    __rmod__: Any = ...
    __divmod__: Any = ...
    __rdivmod__: Any = ...
    __truediv__: Any = ...
    __rtruediv__: Any = ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Any:
        ...

    def join(
            self, other: Any, how: str = ...,
            level: Optional[Any] = ..., return_indexers: bool = ...,
            sort: bool = ...) -> Any:
        ...

    def intersection(self, other: Any, sort: bool = ...) -> Any:
        ...

    def get_value(self, series: Any, key: Any) -> Any:
        ...

    def get_value_maybe_box(self, series: Any, key: Any) -> Any:
        ...

    def get_loc(
            self, key: Any, method: Optional[Any] = ...,
            tolerance: Optional[Any] = ...) -> Any:
        ...

    def searchsorted(
            self, value: Any, side: str = ...,
            sorter: Optional[Any] = ...) -> Any:
        ...

    def is_type_compatible(self, typ: Any) -> Any:
        ...

    @property
    def inferred_type(self) -> Any:
        ...

    @property
    def is_all_dates(self) -> Any:
        ...

    def insert(self, loc: Any, item: Any) -> Any:
        ...

    def delete(self, loc: Any) -> Any:
        ...


def timedelta_range(
        start: Optional[Any] = ..., end: Optional[Any] = ...,
        periods: Optional[Any] = ..., freq: Optional[Any] = ...,
        name: Optional[Any] = ...,
        closed: Optional[Any] = ...) -> Any:
    ...
