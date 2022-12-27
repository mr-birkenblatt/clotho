# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch.autograd.function import Function as Function


class SyncBatchNorm(Function):
    process_group: Incomplete

    @staticmethod
    def forward(
        self, input, weight, bias, running_mean, running_var, eps, momentum,
        process_group, world_size): ...

    @staticmethod
    def backward(self, grad_output): ...


class CrossMapLRN2d(Function):

    @staticmethod
    def forward(
        ctx, input, size, alpha: float = ..., beta: float = ...,
        k: int = ...): ...

    @staticmethod
    def backward(ctx, grad_output): ...


class BackwardHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args): ...
    @staticmethod
    def backward(ctx, *args): ...
