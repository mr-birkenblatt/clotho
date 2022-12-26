from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.exponential import Exponential as Exponential
from torch.distributions.gumbel import euler_constant as euler_constant
from torch.distributions.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from torch.distributions.transforms import AffineTransform as AffineTransform
from torch.distributions.transforms import PowerTransform as PowerTransform
from torch.distributions.utils import broadcast_all as broadcast_all


class Weibull(TransformedDistribution):
    arg_constraints: Incomplete
    support: Incomplete
    concentration_reciprocal: Incomplete
    def __init__(self, scale, concentration, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def entropy(self): ...
