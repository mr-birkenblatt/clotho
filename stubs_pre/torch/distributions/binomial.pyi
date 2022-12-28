# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import broadcast_all as broadcast_all
from torch.distributions.utils import lazy_property as lazy_property
from torch.distributions.utils import logits_to_probs as logits_to_probs
from torch.distributions.utils import probs_to_logits as probs_to_logits


class Binomial(Distribution):
    arg_constraints: Incomplete
    has_enumerate_support: bool
    total_count: Incomplete

    def __init__(
        self, total_count: int = ..., probs: Incomplete | None = ...,
        logits: Incomplete | None = ...,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def support(self): ...
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
    def entropy(self): ...
    def enumerate_support(self, expand: bool = ...): ...
