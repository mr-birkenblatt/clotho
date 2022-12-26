from _typeshed import Incomplete

from .optimizer import Optimizer as Optimizer


class SparseAdam(Optimizer):
    def __init__(self, params, lr: float = ..., betas=..., eps: float = ...) -> None: ...
    def step(self, closure: Incomplete | None = ...): ...
