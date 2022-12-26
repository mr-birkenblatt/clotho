from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class Rprop(Optimizer):
    def __init__(self, params, lr: float = ..., etas=..., step_sizes=..., foreach: Optional[bool] = ...) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...

def rprop(params: List[Tensor], grads: List[Tensor], prevs: List[Tensor], step_sizes: List[Tensor], foreach: bool = ..., *, step_size_min: float, step_size_max: float, etaminus: float, etaplus: float): ...
