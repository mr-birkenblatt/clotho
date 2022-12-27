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
from torch.distributions.exp_family import as, ExponentialFamily


        ExponentialFamily
from torch.distributions.utils import broadcast_all as broadcast_all


        clamp_probs as clamp_probs, lazy_property as lazy_property,
        logits_to_probs as logits_to_probs, probs_to_logits as probs_to_logits
from torch.nn.functional import as, binary_cross_entropy_with_logits


        binary_cross_entropy_with_logits


class ContinuousBernoulli(ExponentialFamily):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool

    def __init__(
        self, probs: Incomplete | None = ...,
        logits: Incomplete | None = ..., lims=...,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def mean(self): ...
    @property
    def stddev(self): ...
    @property
    def variance(self): ...
    def logits(self): ...
    def probs(self): ...
    @property
    def param_shape(self): ...
    def sample(self, sample_shape=...): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    def cdf(self, value): ...
    def icdf(self, value): ...
    def entropy(self): ...
