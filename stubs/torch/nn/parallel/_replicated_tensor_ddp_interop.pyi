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
