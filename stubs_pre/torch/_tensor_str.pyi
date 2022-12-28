# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from typing import Optional

from _typeshed import Incomplete
from torch._six import inf as inf


class __PrinterOptions:
    precision: int
    threshold: float
    edgeitems: int
    linewidth: int
    sci_mode: Optional[bool]

PRINT_OPTS: Incomplete


def set_printoptions(
    precision: Incomplete | None = ..., threshold: Incomplete | None = ...,
    edgeitems: Incomplete | None = ..., linewidth: Incomplete | None = ...,
    profile: Incomplete | None = ...,
    sci_mode: Incomplete | None = ...) -> None: ...


def tensor_totype(t): ...


class _Formatter:
    floating_dtype: Incomplete
    int_mode: bool
    sci_mode: bool
    max_width: int
    def __init__(self, tensor) -> None: ...
    def width(self): ...
    def format(self, value): ...


def get_summarized_data(self): ...
