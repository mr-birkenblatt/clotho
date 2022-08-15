# Stubs for pandas.core.arrays.categorical (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,super-init-not-called,redefined-outer-name
# pylint: disable=no-self-use,arguments-differ,too-many-arguments
# pylint: disable=dangerous-default-value,too-many-ancestors,invalid-name
# pylint: disable=signature-differs,keyword-arg-before-vararg

from typing import Any, Optional
from pandas.core.arrays.base import ExtensionArray
from pandas.core.accessor import PandasDelegate
from pandas.core.base import NoNewAttributesMixin, PandasObject
from pandas.core.dtypes.dtypes import CategoricalDtype


def contains(cat: Any, key: Any, container: Any) -> Any:
    ...


class Categorical(ExtensionArray, PandasObject):
    __array_priority__: int = ...

    def __init__(
            self, values: Any, categories: Optional[Any] = ...,
            ordered: Optional[Any] = ..., dtype: Optional[Any] = ...,
            fastpath: bool = ...) -> None:
        ...

    @property
    def categories(self) -> Any:
        ...

    @categories.setter
    def categories(self, categories: Any) -> None:
        ...

    @property
    def ordered(self) -> Any:
        ...

    @property
    def dtype(self) -> CategoricalDtype:
        ...

    def copy(self) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Any:
        ...

    def ndim(self) -> Any:
        ...

    def size(self) -> Any:
        ...

    def itemsize(self) -> Any:
        ...

    def tolist(self) -> Any:
        ...

    to_list: Any = ...

    @property
    def base(self) -> None:
        ...

    @classmethod
    def from_codes(
            cls, codes: Any, categories: Optional[Any] = ...,
            ordered: Optional[Any] = ...,
            dtype: Optional[Any] = ...) -> Any:
        ...

    codes: Any = ...

    def set_ordered(self, value: Any, inplace: bool = ...) -> Any:
        ...

    def as_ordered(self, inplace: bool = ...) -> Any:
        ...

    def as_unordered(self, inplace: bool = ...) -> Any:
        ...

    def set_categories(
            self, new_categories: Any, ordered: Optional[Any] = ...,
            rename: bool = ..., inplace: bool = ...) -> Any:
        ...

    def rename_categories(
            self, new_categories: Any,
            inplace: bool = ...) -> Any:
        ...

    def reorder_categories(
            self, new_categories: Any,
            ordered: Optional[Any] = ...,
            inplace: bool = ...) -> Any:
        ...

    def add_categories(self, new_categories: Any, inplace: bool = ...) -> Any:
        ...

    def remove_categories(self, removals: Any, inplace: bool = ...) -> Any:
        ...

    def remove_unused_categories(self, inplace: bool = ...) -> Any:
        ...

    def map(self, mapper: Any) -> Any:
        ...

    __eq__: Any = ...
    __ne__: Any = ...
    __lt__: Any = ...
    __gt__: Any = ...
    __le__: Any = ...
    __ge__: Any = ...

    @property
    def shape(self) -> Any:
        ...

    def __array__(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def __array_ufunc__(
            self, ufunc: Any, method: Any, *inputs: Any,
            **kwargs: Any) -> Any:
        ...

    @property
    def T(self) -> Any:
        ...

    @property
    def nbytes(self) -> Any:
        ...

    def memory_usage(self, deep: bool = ...) -> Any:
        ...

    def searchsorted(
            self, value: Any, side: str = ...,
            sorter: Optional[Any] = ...) -> Any:
        ...

    def isna(self) -> Any:
        ...

    isnull: Any = ...

    def notna(self) -> Any:
        ...

    notnull: Any = ...

    def put(self, *args: Any, **kwargs: Any) -> None:
        ...

    def dropna(self) -> Any:
        ...

    def value_counts(self, dropna: bool = ...) -> Any:
        ...

    def get_values(self) -> Any:
        ...

    def check_for_ordered(self, op: Any) -> None:
        ...

    def argsort(
            self, ascending: bool = ..., kind: str = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def sort_values(
            self, inplace: bool = ..., ascending: bool = ...,
            na_position: str = ...) -> Any:
        ...

    def ravel(self, order: str = ...) -> Any:
        ...

    def view(self) -> Any:
        ...

    def to_dense(self) -> Any:
        ...

    def fillna(
            self, value: Optional[Any] = ..., method: Optional[Any] = ...,
            limit: Optional[Any] = ...) -> Any:
        ...

    def take_nd(
            self, indexer: Any, allow_fill: Optional[Any] = ...,
            fill_value: Optional[Any] = ...) -> Any:
        ...

    take: Any = ...

    def __len__(self) -> Any:
        ...

    def __iter__(self) -> Any:
        ...

    def __contains__(self, key: Any) -> Any:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def min(self, numeric_only: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def max(self, numeric_only: Optional[Any] = ..., **kwargs: Any) -> Any:
        ...

    def mode(self, dropna: bool = ...) -> Any:
        ...

    def unique(self) -> Any:
        ...

    def equals(self, other: Any) -> Any:
        ...

    def is_dtype_equal(self, other: Any) -> Any:
        ...

    def describe(self) -> Any:
        ...

    def repeat(self, repeats: Any, axis: Optional[Any] = ...) -> Any:
        ...

    def isin(self, values: Any) -> Any:
        ...


class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: Any) -> None:
        ...

    @property
    def codes(self) -> Any:
        ...

    @property
    def categorical(self) -> Any:
        ...

    @property
    def name(self) -> Any:
        ...

    @property
    def index(self) -> Any:
        ...
