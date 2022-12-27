# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Tuple

from _typeshed import Incomplete
from torch import Size as Size
from torch import Tensor as Tensor
from torch.nn.parameter import Parameter as Parameter

from .. import init as init
from .module import Module as Module


class LocalResponseNorm(Module):
    __constants__: Incomplete
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = ..., beta: float = ...,
        k: float = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...


class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = ..., beta: float = ...,
        k: float = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class LayerNorm(Module):
    __constants__: Incomplete
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, normalized_shape: _shape_t, eps: float = ...,
        elementwise_affine: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class GroupNorm(Module):
    __constants__: Incomplete
    num_groups: int
    num_channels: int
    eps: float
    affine: bool
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = ...,
        affine: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
