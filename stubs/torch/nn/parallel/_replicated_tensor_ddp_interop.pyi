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
from torch.distributed._shard.replicated_tensor import (
    ReplicatedTensor as ReplicatedTensor,
)


class ReplicatedTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, process_group: Incomplete | None = ...): ...
    @staticmethod
    def backward(ctx, grad_output): ...
