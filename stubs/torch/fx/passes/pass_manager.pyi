from typing import Callable, List

from _typeshed import Incomplete


def inplace_wrapper(fn: Callable) -> Callable: ...
def loop_pass(base_pass: Callable, n_iter: int = ..., predicate: Callable = ...): ...
def this_before_that_pass_constraint(this: Callable, that: Callable): ...
def these_before_those_pass_constraint(these: Callable, those: Callable): ...

class PassManager:
    passes: List[Callable]
    constraints: List[Callable]
    def __init__(self, passes: Incomplete | None = ..., constraints: Incomplete | None = ...) -> None: ...
    @classmethod
    def build_from_passlist(cls, passes): ...
    def add_pass(self, _pass: Callable): ...
    def add_constraint(self, constraint) -> None: ...
    def validate(self) -> None: ...
    def __call__(self, source): ...