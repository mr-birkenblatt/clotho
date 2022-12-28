# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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
