from _typeshed import Incomplete
from torch.distributions import Beta as Beta
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import broadcast_all as broadcast_all


class LKJCholesky(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    dim: Incomplete
    def __init__(self, dim, concentration: float = ..., validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
