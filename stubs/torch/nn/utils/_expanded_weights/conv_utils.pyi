# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .expanded_weights_utils import as, set_grad_sample_if_exists


        set_grad_sample_if_exists,
        unpack_expanded_weight_or_tensor as unpack_expanded_weight_or_tensor
from _typeshed import Incomplete


THRESHOLD: int


def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt): ...


def conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs): ...


def conv_normalizer(
    input, weight, bias: Incomplete | None = ..., stride: int = ...,
    padding: int = ..., dilation: int = ..., groups: int = ...): ...


def conv_backward(func, ctx, grad_output): ...


def conv_unfold_weight_grad_sample(
    input, grad_output, weight_shape, kernel_size, stride, padding, dilation,
    groups, func): ...


def conv_group_weight_grad_sample(
    input, grad_output, weight_shape, stride, padding, dilation, batch_size,
    func): ...


def unfold3d(tensor, kernel_size, padding, stride, dilation): ...
