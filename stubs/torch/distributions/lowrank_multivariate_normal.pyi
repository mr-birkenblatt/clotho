# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.utils import lazy_property as lazy_property


class LowRankMultivariateNormal(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    loc: Incomplete
    cov_diag: Incomplete

    def __init__(
        self, loc, cov_factor, cov_diag,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    def variance(self): ...
    def scale_tril(self): ...
    def covariance_matrix(self): ...
    def precision_matrix(self): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
