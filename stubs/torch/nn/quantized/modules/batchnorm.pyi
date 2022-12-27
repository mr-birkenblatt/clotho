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
from torch import Tensor as Tensor


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):

    def __init__(
        self, num_features, eps: float = ..., momentum: float = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    @staticmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point): ...


class BatchNorm2d(_BatchNorm):

    def __init__(
        self, num_features, eps: float = ..., momentum: float = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    @classmethod
    def from_float(cls, mod): ...


class BatchNorm3d(_BatchNorm):

    def __init__(
        self, num_features, eps: float = ..., momentum: float = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    @classmethod
    def from_float(cls, mod): ...
