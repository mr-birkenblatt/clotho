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


class VonMises(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool

    def __init__(
        self, loc, concentration,
        validate_args: Incomplete | None = ...) -> None: ...

    def log_prob(self, value): ...
    def sample(self, sample_shape=...): ...
    def expand(self, batch_shape): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    def variance(self): ...
