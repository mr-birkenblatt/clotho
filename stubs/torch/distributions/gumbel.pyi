from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from torch.distributions.transforms import AffineTransform as AffineTransform
from torch.distributions.transforms import ExpTransform as ExpTransform
from torch.distributions.uniform import Uniform as Uniform
from torch.distributions.utils import broadcast_all as broadcast_all
from torch.distributions.utils import euler_constant as euler_constant


class Gumbel(TransformedDistribution):
    arg_constraints: Incomplete
    support: Incomplete
    def __init__(self, loc, scale, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def log_prob(self, value): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def stddev(self): ...
    @property
    def variance(self): ...
    def entropy(self): ...
