# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import ast
from typing import Any, List, NamedTuple, Optional, Tuple

from _typeshed import Incomplete
from torch._C import ErrorReport as ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory as SourceRangeFactory


def get_source_lines_and_file(
    obj: Any, error_msg: Optional[str] = ...) -> Tuple[List[
                str], int, Optional[str]]: ...


def normalize_source_lines(sourcelines: List[str]) -> List[str]: ...


class SourceContext(SourceRangeFactory):
    uses_true_division: Incomplete
    filename: Incomplete
    funcname: Incomplete

    def __init__(
        self, source, filename, file_lineno, leading_whitespace_len,
        uses_true_division: bool = ...,
        funcname: Incomplete | None = ...) -> None: ...


def make_source_context(*args): ...


def fake_range(): ...


class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: Optional[str]
    file_lineno: int


def parse_def(fn): ...
