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
from torch.distributions.categorical import Categorical as Categorical
from torch.distributions.distribution import Distribution as Distribution
from torch.distributions.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from torch.distributions.transforms import ExpTransform as ExpTransform
from torch.distributions.utils import broadcast_all as broadcast_all
from torch.distributions.utils import clamp_probs as clamp_probs


class ExpRelaxedCategorical(Distribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    temperature: Incomplete

    def __init__(
        self, temperature, probs: Incomplete | None = ...,
        logits: Incomplete | None = ...,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def param_shape(self): ...
    @property
    def logits(self): ...
    @property
    def probs(self): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...


class RelaxedOneHotCategorical(TransformedDistribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool

    def __init__(
        self, temperature, probs: Incomplete | None = ...,
        logits: Incomplete | None = ...,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def temperature(self): ...
    @property
    def logits(self): ...
    @property
    def probs(self): ...
