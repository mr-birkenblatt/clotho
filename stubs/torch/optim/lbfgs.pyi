from _typeshed import Incomplete

from .optimizer import Optimizer as Optimizer


class LBFGS(Optimizer):
    def __init__(self, params, lr: int = ..., max_iter: int = ..., max_eval: Incomplete | None = ..., tolerance_grad: float = ..., tolerance_change: float = ..., history_size: int = ..., line_search_fn: Incomplete | None = ...) -> None: ...
    def step(self, closure): ...
