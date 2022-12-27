# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Dict

from _typeshed import Incomplete
from torch.distributions import constraints as constraints
from torch.distributions.distribution import Distribution as Distribution


class Independent(Distribution):
    arg_constraints: Dict[str, constraints.Constraint]
    base_dist: Incomplete
    reinterpreted_batch_ndims: Incomplete

    def __init__(
        self, base_distribution, reinterpreted_batch_ndims,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def has_rsample(self): ...
    @property
    def has_enumerate_support(self): ...
    def support(self): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def sample(self, sample_shape=...): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def entropy(self): ...
    def enumerate_support(self, expand: bool = ...): ...
