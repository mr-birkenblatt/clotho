from _typeshed import Incomplete
from torch import Tensor as Tensor

from .module import Module as Module


class ChannelShuffle(Module):
    __constants__: Incomplete
    groups: int
    def __init__(self, groups: int) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
