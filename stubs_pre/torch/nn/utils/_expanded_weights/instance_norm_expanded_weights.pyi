# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch

from .expanded_weights_impl import (
    implements_per_sample_grads as implements_per_sample_grads,
)
from .expanded_weights_utils import forward_helper as forward_helper
from .expanded_weights_utils import (
    set_grad_sample_if_exists as set_grad_sample_if_exists,
)
from .expanded_weights_utils import standard_kwargs as standard_kwargs
from .expanded_weights_utils import (
    unpack_expanded_weight_or_tensor as unpack_expanded_weight_or_tensor,
)


class InstanceNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs): ...
    @staticmethod
    def backward(ctx, grad_output): ...
