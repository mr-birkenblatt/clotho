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
from torch.distributions.normal import Normal as Normal
from torch.distributions.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from torch.distributions.transforms import ExpTransform as ExpTransform


class LogNormal(TransformedDistribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool

    def __init__(
        self, loc, scale, validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def loc(self): ...
    @property
    def scale(self): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def entropy(self): ...
