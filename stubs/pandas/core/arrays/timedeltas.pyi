# Stubs for pandas.core.arrays.timedeltas (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,super-init-not-called,redefined-outer-name
# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=dangerous-default-value,too-many-ancestors,inconsistent-mro

from typing import Any, Optional

from pandas.core.arrays import datetimelike as dtl


class TimedeltaArray(dtl.DatetimeLikeArrayMixin, dtl.TimelikeOps):
    __array_priority__: int = ...
    ndim: int = ...

    def __init__(
            self, values: Any, dtype: Any = ..., freq: Optional[Any] = ...,
            copy: bool = ...) -> None:
        ...

    @property
    def dtype(self) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Any:
        ...

    def __mul__(self, other: Any) -> Any:
        ...

    __rmul__: Any = ...

    def __truediv__(self, other: Any) -> Any:
        ...

    def __rtruediv__(self, other: Any) -> Any:
        ...

    def __floordiv__(self, other: Any) -> Any:
        ...

    def __rfloordiv__(self, other: Any) -> Any:
        ...

    def __mod__(self, other: Any) -> Any:
        ...

    def __rmod__(self, other: Any) -> Any:
        ...

    def __divmod__(self, other: Any) -> Any:
        ...

    def __rdivmod__(self, other: Any) -> Any:
        ...

    def __neg__(self) -> Any:
        ...

    def __abs__(self) -> Any:
        ...

    def total_seconds(self) -> Any:
        ...

    def to_pytimedelta(self) -> Any:
        ...

    days: Any = ...
    seconds: Any = ...
    microseconds: Any = ...
    nanoseconds: Any = ...

    @property
    def components(self) -> Any:
        ...


def sequence_to_td64ns(
        data: Any, copy: bool = ..., unit: str = ...,
        errors: str = ...) -> Any:
    ...


def ints_to_td64ns(data: Any, unit: str = ...) -> Any:
    ...


def objects_to_td64ns(data: Any, unit: str = ..., errors: str = ...) -> Any:
    ...
