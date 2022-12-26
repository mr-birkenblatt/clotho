from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.normal import Normal as Normal
from torch.distributions.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from torch.distributions.transforms import (
    StickBreakingTransform as StickBreakingTransform,
)


class LogisticNormal(TransformedDistribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    def __init__(self, loc, scale, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def loc(self): ...
    @property
    def scale(self): ...
