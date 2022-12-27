# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.autograd import Function as Function
from torch.autograd.function import once_differentiable as once_differentiable
from torch.distributions import constraints as constraints
from torch.distributions.exp_family import (
    ExponentialFamily as ExponentialFamily,
)


class _Dirichlet(Function):
    @staticmethod
    def forward(ctx, concentration): ...
    @staticmethod
    def backward(ctx, grad_output): ...


class Dirichlet(ExponentialFamily):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    concentration: Incomplete

    def __init__(
        self, concentration,
        validate_args: Incomplete | None = ...) -> None: ...

    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def rsample(self, sample_shape=...): ...
    def log_prob(self, value): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def entropy(self): ...
