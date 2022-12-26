from typing import Dict

from _typeshed import Incomplete
from torch.distributions import Categorical as Categorical
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution


class MixtureSameFamily(Distribution):
    arg_constraints: Dict[str, constraints.Constraint]
    has_rsample: bool
    def __init__(self, mixture_distribution, component_distribution, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def support(self): ...
    @property
    def mixture_distribution(self): ...
    @property
    def component_distribution(self): ...
    @property
    def mean(self): ...
    @property
    def variance(self): ...
    def cdf(self, x): ...
    def log_prob(self, x): ...
    def sample(self, sample_shape=...): ...
