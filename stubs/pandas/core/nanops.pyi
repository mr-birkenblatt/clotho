# Stubs for pandas.core.nanops (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member

from typing import Any, Optional

bn: Any


def set_use_bottleneck(v: bool = ...) -> None:
    ...


class disallow:
    dtypes: Any = ...

    def __init__(self, *dtypes: Any) -> None:
        ...

    def check(self, obj: Any) -> Any:
        ...

    def __call__(self, f: Any) -> Any:
        ...


class bottleneck_switch:
    name: Any = ...
    kwargs: Any = ...

    def __init__(self, name: Optional[Any] = ..., **kwargs: Any) -> None:
        ...

    def __call__(self, alt: Any) -> Any:
        ...


def nanany(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nanall(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nansum(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        min_count: int = ..., mask: Optional[Any] = ...) -> Any:
    ...


def nanmean(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nanmedian(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nanstd(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        ddof: int = ..., mask: Optional[Any] = ...) -> Any:
    ...


def nanvar(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        ddof: int = ..., mask: Optional[Any] = ...) -> Any:
    ...


def nansem(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        ddof: int = ..., mask: Optional[Any] = ...) -> Any:
    ...


nanmin: Any
nanmax: Any


def nanargmax(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nanargmin(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nanskew(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nankurt(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        mask: Optional[Any] = ...) -> Any:
    ...


def nanprod(
        values: Any, axis: Optional[Any] = ..., skipna: bool = ...,
        min_count: int = ..., mask: Optional[Any] = ...) -> Any:
    ...


def nancorr(
        a: Any, b: Any, method: str = ...,
        min_periods: Optional[Any] = ...) -> Any:
    ...


def get_corr_func(method: Any) -> Any:
    ...


def nancov(a: Any, b: Any, min_periods: Optional[Any] = ...) -> Any:
    ...


def make_nancomp(op: Any) -> Any:
    ...


nangt: Any
nange: Any
nanlt: Any
nanle: Any
naneq: Any
nanne: Any


def nanpercentile(
        values: Any, q: Any, axis: Any, na_value: Any, mask: Any,
        ndim: Any, interpolation: Any) -> Any:
    ...
