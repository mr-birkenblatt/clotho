from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.gamma import Gamma as Gamma


class Chi2(Gamma):
    arg_constraints: Incomplete
    def __init__(self, df, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def df(self): ...
