from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class ASGD(Optimizer):
    def __init__(self, params, lr: float = ..., lambd: float = ..., alpha: float = ..., t0: float = ..., weight_decay: int = ..., foreach: Optional[bool] = ...) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...

def asgd(params: List[Tensor], grads: List[Tensor], axs: List[Tensor], mus: List[Tensor], etas: List[Tensor], state_steps: List[Tensor], foreach: bool = ..., *, lambd: float, lr: float, t0: float, alpha: float, weight_decay: float): ...
