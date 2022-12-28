# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch import Tensor as Tensor

from .module import Module as Module


class _DropoutNd(Module):
    __constants__: Incomplete
    p: float
    inplace: bool
    def __init__(self, p: float = ..., inplace: bool = ...) -> None: ...
    def extra_repr(self) -> str: ...


class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...


class Dropout1d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...


class Dropout2d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...


class Dropout3d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...


class AlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...


class FeatureAlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor: ...
