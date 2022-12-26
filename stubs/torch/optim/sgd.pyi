from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer
from .optimizer import required as required


class SGD(Optimizer):
    def __init__(self, params, lr=..., momentum: int = ..., dampening: int = ..., weight_decay: int = ..., nesterov: bool = ..., *, maximize: bool = ..., foreach: Optional[bool] = ...) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...

def sgd(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]], has_sparse_grad: bool = ..., foreach: bool = ..., *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool): ...
