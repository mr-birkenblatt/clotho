# Stubs for pandas.core.arrays.datetimelike (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,no-self-use,invalid-name
# pylint: disable=relative-beyond-top-level,line-too-long,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin

from typing import Any, Optional
from pandas._libs.tslibs import NaTType
import numpy as np
from .base import ExtensionArray, ExtensionOpsMixin


class AttributesMixin:
    ...


class DatelikeOps:
    def strftime(self, date_format: Any) -> Any:
        ...


class TimelikeOps:
    def round(
            self, freq: Any, ambiguous: str = ...,
            nonexistent: str = ...) -> Any:
        ...

    def floor(
            self, freq: Any, ambiguous: str = ...,
            nonexistent: str = ...) -> Any:
        ...

    def ceil(
            self, freq: Any, ambiguous: str = ...,
            nonexistent: str = ...) -> Any:
        ...


class DatetimeLikeArrayMixin(
        ExtensionOpsMixin,
        AttributesMixin,
        ExtensionArray):
    def __iter__(self) -> Any:
        ...

    @property
    def asi8(self) -> np.ndarray:
        ...

    @property
    def nbytes(self) -> Any:
        ...

    def __array__(self, dtype: Optional[Any] = ...) -> Any:
        ...

    @property
    def size(self) -> int:
        ...

    def __len__(self) -> Any:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Any:
        ...

    def view(self, dtype: Optional[Any] = ...) -> Any:
        ...

    def unique(self) -> Any:
        ...

    def take(
            self, indices: Any, allow_fill: bool = ...,
            fill_value: Optional[Any] = ...) -> Any:
        ...

    def copy(self) -> Any:
        ...

    def searchsorted(
            self, value: Any, side: str = ...,
            sorter: Optional[Any] = ...) -> Any:
        ...

    def repeat(self, repeats: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def value_counts(self, dropna: bool = ...) -> Any:
        ...

    def map(self, mapper: Any) -> Any:
        ...

    def isna(self) -> Any:
        ...

    def fillna(
            self, value: Optional[Any] = ..., method: Optional[Any] = ...,
            limit: Optional[Any] = ...) -> Any:
        ...

    @property
    def freq(self) -> Any:
        ...

    @freq.setter
    def freq(self, value: Any) -> None:
        ...

    @property
    def freqstr(self) -> Any:
        ...

    @property
    def inferred_freq(self) -> Any:
        ...

    @property
    def resolution(self) -> Any:
        ...

    def __add__(self, other: Any) -> Any:
        ...

    def __radd__(self, other: Any) -> Any:
        ...

    def __sub__(self, other: Any) -> Any:
        ...

    def __rsub__(self, other: Any) -> Any:
        ...

    def __iadd__(self, other: Any) -> Any:
        ...

    def __isub__(self, other: Any) -> Any:
        ...

    def min(
            self, axis: Optional[Any] = ..., skipna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def max(
            self, axis: Optional[Any] = ..., skipna: bool = ..., *args: Any,
            **kwargs: Any) -> Any:
        ...

    def mean(self, skipna: bool = ...) -> Any:
        ...


def validate_periods(periods: Any) -> Any:
    ...


def validate_endpoints(closed: Any) -> Any:
    ...


def validate_inferred_freq(
        freq: Any, inferred_freq: Any,
        freq_infer: Any) -> Any:
    ...


def maybe_infer_freq(freq: Any) -> Any:
    ...
