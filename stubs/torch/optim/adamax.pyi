from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class Adamax(Optimizer):
    def __init__(self, params, lr: float = ..., betas=..., eps: float = ..., weight_decay: int = ..., foreach: Optional[bool] = ..., *, maximize: bool = ...) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...

def adamax(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_infs: List[Tensor], state_steps: List[Tensor], foreach: bool = ..., maximize: bool = ..., *, eps: float, beta1: float, beta2: float, lr: float, weight_decay: float): ...
