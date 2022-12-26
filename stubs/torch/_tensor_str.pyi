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

def set_printoptions(precision: Incomplete | None = ..., threshold: Incomplete | None = ..., edgeitems: Incomplete | None = ..., linewidth: Incomplete | None = ..., profile: Incomplete | None = ..., sci_mode: Incomplete | None = ...) -> None: ...
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
