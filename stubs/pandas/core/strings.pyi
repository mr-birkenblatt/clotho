# Stubs for pandas.core.strings (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,no-self-use,invalid-name
# pylint: disable=relative-beyond-top-level,line-too-long,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member

from typing import Any, List, Optional
from pandas.core.base import NoNewAttributesMixin


def cat_core(list_of_columns: List, sep: str) -> Any:
    ...


def cat_safe(list_of_columns: List, sep: str) -> Any:
    ...


def str_count(arr: Any, pat: Any, flags: int = ...) -> Any:
    ...


def str_contains(
        arr: Any, pat: Any, case: bool = ..., flags: int = ...,
        na: Any = ..., regex: bool = ...) -> Any:
    ...


def str_startswith(arr: Any, pat: Any, na: Any = ...) -> Any:
    ...


def str_endswith(arr: Any, pat: Any, na: Any = ...) -> Any:
    ...


def str_replace(
        arr: Any, pat: Any, repl: Any, n: int = ...,
        case: Optional[Any] = ..., flags: int = ...,
        regex: bool = ...) -> Any:
    ...


def str_repeat(arr: Any, repeats: Any) -> Any:
    ...


def str_match(
        arr: Any, pat: Any, case: bool = ..., flags: int = ...,
        na: Any = ...) -> Any:
    ...


def str_extract(
        arr: Any, pat: Any, flags: int = ...,
        expand: bool = ...) -> Any:
    ...


def str_extractall(arr: Any, pat: Any, flags: int = ...) -> Any:
    ...


def str_get_dummies(arr: Any, sep: str = ...) -> Any:
    ...


def str_join(arr: Any, sep: Any) -> Any:
    ...


def str_findall(arr: Any, pat: Any, flags: int = ...) -> Any:
    ...


def str_find(
        arr: Any, sub: Any, start: int = ..., end: Optional[Any] = ...,
        side: str = ...) -> Any:
    ...


def str_index(
        arr: Any, sub: Any, start: int = ..., end: Optional[Any] = ...,
        side: str = ...) -> Any:
    ...


def str_pad(
        arr: Any, width: Any, side: str = ...,
        fillchar: str = ...) -> Any:
    ...


def str_split(
        arr: Any, pat: Optional[Any] = ...,
        n: Optional[Any] = ...) -> Any:
    ...


def str_rsplit(
        arr: Any, pat: Optional[Any] = ...,
        n: Optional[Any] = ...) -> Any:
    ...


def str_slice(
        arr: Any, start: Optional[Any] = ..., stop: Optional[Any] = ...,
        step: Optional[Any] = ...) -> Any:
    ...


def str_slice_replace(
        arr: Any, start: Optional[Any] = ...,
        stop: Optional[Any] = ...,
        repl: Optional[Any] = ...) -> Any:
    ...


def str_strip(arr: Any, to_strip: Optional[Any] = ..., side: str = ...) -> Any:
    ...


def str_wrap(arr: Any, width: Any, **kwargs: Any) -> Any:
    ...


def str_translate(arr: Any, table: Any) -> Any:
    ...


def str_get(arr: Any, i: Any) -> Any:
    ...


def str_decode(arr: Any, encoding: Any, errors: str = ...) -> Any:
    ...


def str_encode(arr: Any, encoding: Any, errors: str = ...) -> Any:
    ...


def forbid_nonstring_types(forbidden: Any, name: Optional[Any] = ...) -> Any:
    ...


def copy(source: Any) -> Any:
    ...


class StringMethods(NoNewAttributesMixin):
    def __init__(self, data: Any) -> None:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __iter__(self) -> None:
        ...

    def cat(
            self, others: Optional[Any] = ..., sep: Optional[Any] = ...,
            na_rep: Optional[Any] = ..., join: Optional[Any] = ...) -> Any:
        ...

    def split(
            self, pat: Optional[Any] = ..., n: int = ...,
            expand: bool = ...) -> Any:
        ...

    def rsplit(
            self, pat: Optional[Any] = ..., n: int = ...,
            expand: bool = ...) -> Any:
        ...

    def partition(self, sep: str = ..., expand: bool = ...) -> Any:
        ...

    def rpartition(self, sep: str = ..., expand: bool = ...) -> Any:
        ...

    def get(self, i: Any) -> Any:
        ...

    def join(self, sep: Any) -> Any:
        ...

    def contains(
            self, pat: Any, case: bool = ..., flags: int = ...,
            na: Any = ..., regex: bool = ...) -> Any:
        ...

    def match(
            self, pat: Any, case: bool = ..., flags: int = ...,
            na: Any = ...) -> Any:
        ...

    def replace(
            self, pat: Any, repl: Any, n: int = ...,
            case: Optional[Any] = ..., flags: int = ...,
            regex: bool = ...) -> Any:
        ...

    def repeat(self, repeats: Any) -> Any:
        ...

    def pad(self, width: Any, side: str = ..., fillchar: str = ...) -> Any:
        ...

    def center(self, width: Any, fillchar: str = ...) -> Any:
        ...

    def ljust(self, width: Any, fillchar: str = ...) -> Any:
        ...

    def rjust(self, width: Any, fillchar: str = ...) -> Any:
        ...

    def zfill(self, width: Any) -> Any:
        ...

    def slice(
            self, start: Optional[Any] = ..., stop: Optional[Any] = ...,
            step: Optional[Any] = ...) -> Any:
        ...

    def slice_replace(
            self, start: Optional[Any] = ...,
            stop: Optional[Any] = ...,
            repl: Optional[Any] = ...) -> Any:
        ...

    def decode(self, encoding: Any, errors: str = ...) -> Any:
        ...

    def encode(self, encoding: Any, errors: str = ...) -> Any:
        ...

    def strip(self, to_strip: Optional[Any] = ...) -> Any:
        ...

    def lstrip(self, to_strip: Optional[Any] = ...) -> Any:
        ...

    def rstrip(self, to_strip: Optional[Any] = ...) -> Any:
        ...

    def wrap(self, width: Any, **kwargs: Any) -> Any:
        ...

    def get_dummies(self, sep: str = ...) -> Any:
        ...

    def translate(self, table: Any) -> Any:
        ...

    count: Any = ...
    startswith: Any = ...
    endswith: Any = ...
    findall: Any = ...

    def extract(self, pat: Any, flags: int = ..., expand: bool = ...) -> Any:
        ...

    def extractall(self, pat: Any, flags: int = ...) -> Any:
        ...

    def find(
            self, sub: Any, start: int = ...,
            end: Optional[Any] = ...) -> Any:
        ...

    def rfind(
            self, sub: Any, start: int = ...,
            end: Optional[Any] = ...) -> Any:
        ...

    def normalize(self, form: Any) -> Any:
        ...

    def index(
            self, sub: Any, start: int = ...,
            end: Optional[Any] = ...) -> Any:
        ...

    def rindex(
            self, sub: Any, start: int = ...,
            end: Optional[Any] = ...) -> Any:
        ...

    len: Any = ...
    lower: Any = ...
    upper: Any = ...
    title: Any = ...
    capitalize: Any = ...
    swapcase: Any = ...
    casefold: Any = ...
    isalnum: Any = ...
    isalpha: Any = ...
    isdigit: Any = ...
    isspace: Any = ...
    islower: Any = ...
    isupper: Any = ...
    istitle: Any = ...
    isnumeric: Any = ...
    isdecimal: Any = ...
