# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch.nn.quantized.functional
from _typeshed import Incomplete


class ReLU6(torch.nn.ReLU):
    inplace: Incomplete
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input): ...
    @staticmethod
    def from_float(mod): ...


class Hardswish(torch.nn.Hardswish):
    scale: Incomplete
    zero_point: Incomplete
    def __init__(self, scale, zero_point) -> None: ...
    def forward(self, input): ...
    @staticmethod
    def from_float(mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...


class ELU(torch.nn.ELU):
    scale: Incomplete
    zero_point: Incomplete
    def __init__(self, scale, zero_point, alpha: float = ...) -> None: ...
    def forward(self, input): ...
    @staticmethod
    def from_float(mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...


class LeakyReLU(torch.nn.LeakyReLU):

    def __init__(
        self, scale: float, zero_point: int, negative_slope: float = ...,
        inplace: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...


class Sigmoid(torch.nn.Sigmoid):
    output_scale: Incomplete
    output_zero_point: Incomplete

    def __init__(
        self, output_scale: float, output_zero_point: int) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class Softmax(torch.nn.Softmax):
    dim: Incomplete
    scale: Incomplete
    zero_point: Incomplete

    def __init__(
        self, dim: Incomplete | None = ..., scale: float = ...,
        zero_point: int = ...) -> None: ...

    def forward(self, input): ...
    @staticmethod
    def from_float(mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...
