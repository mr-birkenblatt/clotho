from typing import Any, Dict, Optional

from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.utils import lazy_property as lazy_property


class Distribution:
    has_rsample: bool
    has_enumerate_support: bool
    @staticmethod
    def set_default_validate_args(value) -> None: ...
    def __init__(self, batch_shape=..., event_shape=..., validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...) -> None: ...
    @property
    def batch_shape(self): ...
    @property
    def event_shape(self): ...
    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]: ...
    @property
    def support(self) -> Optional[Any]: ...
    @property
    def mean(self) -> None: ...
    @property
    def mode(self) -> None: ...
    @property
    def variance(self) -> None: ...
    @property
    def stddev(self): ...
    def sample(self, sample_shape=...): ...
    def rsample(self, sample_shape=...) -> None: ...
    def sample_n(self, n): ...
    def log_prob(self, value) -> None: ...
    def cdf(self, value) -> None: ...
    def icdf(self, value) -> None: ...
    def enumerate_support(self, expand: bool = ...) -> None: ...
    def entropy(self) -> None: ...
    def perplexity(self): ...