# Stubs for pandas.core.indexes.category (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,no-self-use,invalid-name
# pylint: disable=relative-beyond-top-level,line-too-long,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs

from typing import Any, Optional
from pandas._typing import AnyArrayLike
from pandas.core import accessor
from pandas.core.indexes.base import Index


class CategoricalIndex(Index, accessor.PandasDelegate):
    def __new__(
            cls, data: Optional[Any] = ...,
            categories: Optional[Any] = ..., ordered: Optional[Any] = ...,
            dtype: Optional[Any] = ..., copy: bool = ...,
            name: Optional[Any] = ...,
            fastpath: Optional[Any] = ...) -> Any:
        ...

    def equals(self, other: Any) -> Any:
        ...

    def inferred_type(self) -> Any:
        ...

    @property
    def values(self) -> Any:
        ...

    @property
    def itemsize(self) -> Any:
        ...

    def tolist(self) -> Any:
        ...

    @property
    def codes(self) -> Any:
        ...

    @property
    def categories(self) -> Any:
        ...

    @property
    def ordered(self) -> Any:
        ...

    def __contains__(self, key: Any) -> Any:
        ...

    def __array__(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Any:
        ...

    def argsort(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def is_unique(self) -> Any:
        ...

    @property
    def is_monotonic_increasing(self) -> Any:
        ...

    @property
    def is_monotonic_decreasing(self) -> Any:
        ...

    def unique(self, level: Optional[Any] = ...) -> Any:
        ...

    def duplicated(self, keep: str = ...) -> Any:
        ...

    def get_value(self, series: AnyArrayLike, key: Any) -> Any:
        ...

    def where(self, cond: Any, other: Optional[Any] = ...) -> Any:
        ...

    def reindex(
            self, target: Any, method: Optional[Any] = ...,
            level: Optional[Any] = ..., limit: Optional[Any] = ...,
            tolerance: Optional[Any] = ...) -> Any:
        ...

    def get_indexer(
            self, target: Any, method: Optional[Any] = ...,
            limit: Optional[Any] = ...,
            tolerance: Optional[Any] = ...) -> Any:
        ...

    def get_indexer_non_unique(self, target: Any) -> Any:
        ...

    def take(
            self, indices: Any, axis: int = ..., allow_fill: bool = ...,
            fill_value: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def is_dtype_equal(self, other: Any) -> Any:
        ...

    take_nd: Any = ...

    def delete(self, loc: Any) -> Any:
        ...

    def insert(self, loc: Any, item: Any) -> Any:
        ...
