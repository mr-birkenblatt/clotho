from torch.autograd import Function as Function

from . import comm as comm


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input): ...
    @staticmethod
    def backward(ctx, *grad_output): ...
