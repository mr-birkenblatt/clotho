# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch._six import inf as inf
from torch.distributions import Categorical as Categorical
from torch.distributions import constraints as constraints
from torch.distributions.binomial import Binomial as Binomial
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import broadcast_all as broadcast_all


class Multinomial(Distribution):
    arg_constraints: Incomplete
    total_count: int
    @property
    def mean(self): ...
    @property
    def variance(self): ...

    def __init__(
        self, total_count: int = ..., probs: Incomplete | None = ...,
        logits: Incomplete | None = ...,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def support(self): ...
    @property
    def logits(self): ...
    @property
    def probs(self): ...
    @property
    def param_shape(self): ...
    def sample(self, sample_shape=...): ...
    def entropy(self): ...
    def log_prob(self, value): ...
