from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr: float = ..., lr_decay: int = ..., weight_decay: int = ..., initial_accumulator_value: int = ..., eps: float = ..., foreach: Optional[bool] = ..., *, maximize: bool = ...) -> None: ...
    def share_memory(self) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...

def adagrad(params: List[Tensor], grads: List[Tensor], state_sums: List[Tensor], state_steps: List[Tensor], has_sparse_grad: bool = ..., foreach: bool = ..., *, lr: float, weight_decay: float, lr_decay: float, eps: float, maximize: bool): ...
