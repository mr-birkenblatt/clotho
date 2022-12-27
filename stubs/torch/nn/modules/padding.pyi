# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Sequence, Tuple

from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..common_types import _size_2_t, _size_4_t, _size_6_t
from .module import Module as Module


class _ConstantPadNd(Module):
    __constants__: Incomplete
    value: float
    padding: Sequence[int]
    def __init__(self, value: float) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class ConstantPad1d(_ConstantPadNd):
    padding: Tuple[int, int]
    def __init__(self, padding: _size_2_t, value: float) -> None: ...


class ConstantPad2d(_ConstantPadNd):
    __constants__: Incomplete
    padding: Tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t, value: float) -> None: ...


class ConstantPad3d(_ConstantPadNd):
    padding: Tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t, value: float) -> None: ...


class _ReflectionPadNd(Module):
    __constants__: Incomplete
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class ReflectionPad1d(_ReflectionPadNd):
    padding: Tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...


class ReflectionPad2d(_ReflectionPadNd):
    padding: Tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...


class ReflectionPad3d(_ReflectionPadNd):
    padding: Tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...


class _ReplicationPadNd(Module):
    __constants__: Incomplete
    padding: Sequence[int]
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class ReplicationPad1d(_ReplicationPadNd):
    padding: Tuple[int, int]
    def __init__(self, padding: _size_2_t) -> None: ...


class ReplicationPad2d(_ReplicationPadNd):
    padding: Tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...


class ReplicationPad3d(_ReplicationPadNd):
    padding: Tuple[int, int, int, int, int, int]
    def __init__(self, padding: _size_6_t) -> None: ...


class ZeroPad2d(ConstantPad2d):
    padding: Tuple[int, int, int, int]
    def __init__(self, padding: _size_4_t) -> None: ...
    def extra_repr(self) -> str: ...
