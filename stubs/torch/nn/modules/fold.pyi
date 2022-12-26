from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..common_types import _size_any_t
from .module import Module as Module


class Fold(Module):
    __constants__: Incomplete
    output_size: _size_any_t
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t
    def __init__(self, output_size: _size_any_t, kernel_size: _size_any_t, dilation: _size_any_t = ..., padding: _size_any_t = ..., stride: _size_any_t = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class Unfold(Module):
    __constants__: Incomplete
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t
    def __init__(self, kernel_size: _size_any_t, dilation: _size_any_t = ..., padding: _size_any_t = ..., stride: _size_any_t = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
