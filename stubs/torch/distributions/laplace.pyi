from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import broadcast_all as broadcast_all


class Laplace(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    @property
    def stddev(self): ...
    def __init__(self, loc, scale, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def cdf(self, value): ...
    def icdf(self, value): ...
    def entropy(self): ...