import torch

from .expanded_weights_impl import (
    implements_per_sample_grads as implements_per_sample_grads,
)
from .expanded_weights_utils import forward_helper as forward_helper
from .expanded_weights_utils import (
    set_grad_sample_if_exists as set_grad_sample_if_exists,
)
from .expanded_weights_utils import (
    unpack_expanded_weight_or_tensor as unpack_expanded_weight_or_tensor,
)


class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _, __, *expanded_args_and_kwargs): ...
    @staticmethod
    def backward(ctx, grad_output): ...