# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any

from _typeshed import Incomplete
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module


class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...


class Linear(Module):
    __constants__: Incomplete
    in_features: int
    out_features: int
    weight: Tensor
    bias: Incomplete

    def __init__(
        self, in_features: int, out_features: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class NonDynamicallyQuantizableLinear(Linear):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class Bilinear(Module):
    __constants__: Incomplete
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor
    bias: Incomplete

    def __init__(
        self, in1_features: int, in2_features: int, out_features: int,
        bias: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    def forward(self, input1: Tensor, input2: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class LazyLinear(LazyModuleMixin, Linear):
    cls_to_become: Incomplete
    weight: UninitializedParameter
    bias: UninitializedParameter
    out_features: Incomplete

    def __init__(
        self, out_features: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    in_features: Incomplete
    def initialize_parameters(self, input) -> None: ...
