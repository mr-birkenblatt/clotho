# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._six import inf as inf
from torch._six import nan as nan
from torch.distributions import Chi2 as Chi2
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import broadcast_all as broadcast_all


class StudentT(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...

    def __init__(
        self, df, loc: float = ..., scale: float = ...,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
