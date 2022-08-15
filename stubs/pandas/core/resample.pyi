# Stubs for pandas.core.resample (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,no-name-in-module,super-init-not-called
from typing import Any, Optional

from pandas.core.groupby.base import GroupByMixin
from pandas.core.groupby.groupby import _GroupBy
from pandas.core.groupby.grouper import Grouper


class Resampler(_GroupBy):
    groupby: Any = ...
    keys: Any = ...
    sort: bool = ...
    axis: Any = ...
    kind: Any = ...
    squeeze: bool = ...
    group_keys: bool = ...
    as_index: bool = ...
    exclusions: Any = ...
    binner: Any = ...
    grouper: Any = ...

    def __init__(
            self, obj: Any, groupby: Optional[Any] = ..., axis: int = ...,
            kind: Optional[Any] = ..., **kwargs: Any) -> None:
        ...

    def __getattr__(self, attr: Any) -> Any:
        ...

    def __iter__(self) -> Any:
        ...

    @property
    def obj(self) -> Any:
        ...

    @property
    def ax(self) -> Any:
        ...

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    agg: Any = ...
    apply: Any = ...

    def pad(self, limit: Optional[Any] = ...) -> Any:
        ...

    ffill: Any = ...

    def nearest(self, limit: Optional[Any] = ...) -> Any:
        ...

    def backfill(self, limit: Optional[Any] = ...) -> Any:
        ...

    bfill: Any = ...

    def fillna(self, method: Any, limit: Optional[Any] = ...) -> Any:
        ...

    def interpolate(
            self, method: str = ..., axis: int = ...,
            limit: Optional[Any] = ..., inplace: bool = ...,
            limit_direction: str = ...,
            limit_area: Optional[Any] = ...,
            downcast: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def asfreq(self, fill_value: Optional[Any] = ...) -> Any:
        ...

    def std(self, ddof: int = ..., **kwargs: Any) -> Any:
        ...

    def var(self, ddof: int = ..., **kwargs: Any) -> Any:
        ...

    def size(self) -> Any:
        ...

    def quantile(self, q: float = ..., **kwargs: Any) -> Any:
        ...


class _GroupByMixin(GroupByMixin):
    groupby: Any = ...

    def __init__(self, obj: Any, *args: Any, **kwargs: Any) -> None:
        ...


class DatetimeIndexResampler(Resampler):
    ...


class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    ...


class PeriodIndexResampler(DatetimeIndexResampler):
    ...


class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    ...


class TimedeltaIndexResampler(DatetimeIndexResampler):
    ...


class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    ...


def resample(obj: Any, kind: Optional[Any] = ..., **kwds: Any) -> Any:
    ...


def get_resampler_for_grouping(
        groupby: Any, rule: Any, how: Optional[Any] = ...,
        fill_method: Optional[Any] = ..., limit: Optional[Any] = ...,
        kind: Optional[Any] = ..., **kwargs: Any) -> Any:
    ...


class TimeGrouper(Grouper):
    closed: Any = ...
    label: Any = ...
    kind: Any = ...
    convention: Any = ...
    loffset: Any = ...
    how: Any = ...
    fill_method: Any = ...
    limit: Any = ...
    base: Any = ...

    def __init__(
            self, freq: str = ..., closed: Optional[Any] = ...,
            label: Optional[Any] = ..., how: str = ..., axis: int = ...,
            fill_method: Optional[Any] = ..., limit: Optional[Any] = ...,
            loffset: Optional[Any] = ..., kind: Optional[Any] = ...,
            convention: Optional[Any] = ..., base: int = ...,
            **kwargs: Any) -> None:
        ...


def asfreq(
        obj: Any, freq: Any, method: Optional[Any] = ...,
        how: Optional[Any] = ..., normalize: bool = ...,
        fill_value: Optional[Any] = ...) -> Any:
    ...
