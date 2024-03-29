# Stubs for pandas.core.groupby.generic (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,no-name-in-module,arguments-differ
# pylint: disable=keyword-arg-before-vararg,abstract-method
# pylint: disable=too-many-ancestors
from collections import namedtuple
from typing import Any, Callable, FrozenSet, Iterator, Optional, Type, Union

from pandas._typing import FrameOrSeries
from pandas.core.groupby.groupby import GroupBy


NamedAgg = namedtuple('NamedAgg', ['column', 'aggfunc'])
AggScalar = Union[str, Callable[..., Any]]
ScalarResult: Any


def whitelist_method_generator(
        base_class: Type[GroupBy],
        klass: Type[FrameOrSeries],
        whitelist: FrozenSet[str]) -> Iterator[str]:
    ...


class NDFrameGroupBy(GroupBy):
    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    agg: Any = ...

    def transform(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def filter(
            self, func: Any, dropna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...


class SeriesGroupBy(GroupBy):
    def apply(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    agg: Any = ...

    def transform(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def filter(
            self, func: Any, dropna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def nunique(self, dropna: bool = ...) -> Any:
        ...

    def describe(self, **kwargs: Any) -> Any:
        ...

    def value_counts(
            self, normalize: bool = ..., sort: bool = ...,
            ascending: bool = ..., bins: Optional[Any] = ...,
            dropna: bool = ...) -> Any:
        ...


class DataFrameGroupBy(NDFrameGroupBy):
    agg: Any = ...

    def nunique(self, dropna: bool = ...) -> Any:
        ...

    boxplot: Any = ...
