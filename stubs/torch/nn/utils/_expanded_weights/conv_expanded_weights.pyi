# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


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
