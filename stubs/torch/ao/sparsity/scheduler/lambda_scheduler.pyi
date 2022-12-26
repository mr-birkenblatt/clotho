from _typeshed import Incomplete

from .base_scheduler import BaseScheduler as BaseScheduler


class LambdaSL(BaseScheduler):
    sparsifier: Incomplete
    sl_lambdas: Incomplete
    def __init__(self, sparsifier, sl_lambda, last_epoch: int = ..., verbose: bool = ...) -> None: ...
    def get_sl(self): ...
