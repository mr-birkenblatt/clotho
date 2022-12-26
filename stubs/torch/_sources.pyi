import ast
from typing import Any, List, NamedTuple, Optional, Tuple

from _typeshed import Incomplete
from torch._C import ErrorReport as ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory as SourceRangeFactory


def get_source_lines_and_file(obj: Any, error_msg: Optional[str] = ...) -> Tuple[List[str], int, Optional[str]]: ...
def normalize_source_lines(sourcelines: List[str]) -> List[str]: ...

class SourceContext(SourceRangeFactory):
    uses_true_division: Incomplete
    filename: Incomplete
    funcname: Incomplete
    def __init__(self, source, filename, file_lineno, leading_whitespace_len, uses_true_division: bool = ..., funcname: Incomplete | None = ...) -> None: ...

def make_source_context(*args): ...
def fake_range(): ...

class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: Optional[str]
    file_lineno: int

def parse_def(fn): ...
