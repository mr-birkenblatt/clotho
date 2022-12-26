from _typeshed import Incomplete
from torch.fx.experimental.unification import Var as Var

from ._compatibility import compatibility as compatibility


class TensorType:
    __origin__: Incomplete
    __args__: Incomplete
    def __init__(self, dim) -> None: ...
    def __eq__(self, other): ...
    @staticmethod
    def __class_getitem__(*args): ...

class _DynType:
    __name__: str
    def __init__(self) -> None: ...
    def __eq__(self, other): ...

Dyn: Incomplete

def is_consistent(t1, t2): ...
def is_more_precise(t1, t2): ...
