from typing import List, NamedTuple, Sequence

from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..functional import log_softmax as log_softmax
from . import Linear as Linear
from . import ModuleList as ModuleList
from . import Sequential as Sequential
from .module import Module as Module


class _ASMoutput(NamedTuple):
    output: Incomplete
    loss: Incomplete

class AdaptiveLogSoftmaxWithLoss(Module):
    in_features: int
    n_classes: int
    cutoffs: List[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList
    shortlist_size: Incomplete
    n_clusters: Incomplete
    head_size: Incomplete
    def __init__(self, in_features: int, n_classes: int, cutoffs: Sequence[int], div_value: float = ..., head_bias: bool = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ...) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput: ...
    def log_prob(self, input: Tensor) -> Tensor: ...
    def predict(self, input: Tensor) -> Tensor: ...
