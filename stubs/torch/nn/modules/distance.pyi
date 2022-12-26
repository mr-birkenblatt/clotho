from _typeshed import Incomplete
from torch import Tensor as Tensor

from .module import Module as Module


class PairwiseDistance(Module):
    __constants__: Incomplete
    norm: float
    eps: float
    keepdim: bool
    def __init__(self, p: float = ..., eps: float = ..., keepdim: bool = ...) -> None: ...
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: ...

class CosineSimilarity(Module):
    __constants__: Incomplete
    dim: int
    eps: float
    def __init__(self, dim: int = ..., eps: float = ...) -> None: ...
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: ...
