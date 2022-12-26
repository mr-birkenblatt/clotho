import torch

from .conv_utils import conv_args_and_kwargs as conv_args_and_kwargs
from .conv_utils import conv_backward as conv_backward
from .expanded_weights_impl import ExpandedWeight as ExpandedWeight
from .expanded_weights_impl import (
    implements_per_sample_grads as implements_per_sample_grads,
)
from .expanded_weights_utils import forward_helper as forward_helper


class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, conv_fn, *expanded_args_and_kwargs): ...
    @staticmethod
    def backward(ctx, grad_output): ...
