from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import broadcast_all as broadcast_all
from torch.distributions.utils import lazy_property as lazy_property
from torch.distributions.utils import logits_to_probs as logits_to_probs
from torch.distributions.utils import probs_to_logits as probs_to_logits


class NegativeBinomial(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    total_count: Incomplete
    def __init__(self, total_count, probs: Incomplete | None = ..., logits: Incomplete | None = ..., validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def logits(self): ...
    def probs(self): ...
    @property
    def param_shape(self): ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
