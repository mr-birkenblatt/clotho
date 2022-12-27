# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


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

    def __init__(
        self, output_size: _size_any_t, kernel_size: _size_any_t,
        dilation: _size_any_t = ..., padding: _size_any_t = ...,
        stride: _size_any_t = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Unfold(Module):
    __constants__: Incomplete
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t

    def __init__(
        self, kernel_size: _size_any_t, dilation: _size_any_t = ...,
        padding: _size_any_t = ..., stride: _size_any_t = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
