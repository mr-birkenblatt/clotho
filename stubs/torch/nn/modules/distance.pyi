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

from .module import Module as Module


class PairwiseDistance(Module):
    __constants__: Incomplete
    norm: float
    eps: float
    keepdim: bool

    def __init__(
        self, p: float = ..., eps: float = ...,
        keepdim: bool = ...) -> None: ...

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: ...


class CosineSimilarity(Module):
    __constants__: Incomplete
    dim: int
    eps: float
    def __init__(self, dim: int = ..., eps: float = ...) -> None: ...
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: ...
