from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.categorical import Categorical as Categorical
from torch.distributions.distribution import Distribution as Distribution


class OneHotCategorical(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_enumerate_support: bool
    def __init__(self, probs: Incomplete | None = ..., logits: Incomplete | None = ..., validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def probs(self): ...
    @property
    def logits(self): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    @property
    def param_shape(self): ...
    def sample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
    def enumerate_support(self, expand: bool = ...): ...

class OneHotCategoricalStraightThrough(OneHotCategorical):
    has_rsample: bool
    def rsample(self, sample_shape=...): ...
