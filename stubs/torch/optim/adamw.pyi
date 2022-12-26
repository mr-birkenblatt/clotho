from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr: float = ..., betas=..., eps: float = ..., weight_decay: float = ..., amsgrad: bool = ..., *, maximize: bool = ..., foreach: Optional[bool] = ..., capturable: bool = ...) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...

def adamw(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], foreach: bool = ..., capturable: bool = ..., *, amsgrad: bool, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, maximize: bool): ...
