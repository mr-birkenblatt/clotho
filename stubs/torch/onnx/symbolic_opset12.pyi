# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx import utils as utils


def einsum_helper(g, equation, tensors): ...


def einsum(g, equation, tensor_list): ...


def outer(g, input, other): ...


def dropout(g, input, p, train): ...


def nll_loss(g, self, target, weight, reduction, ignore_index): ...


def nll_loss2d(g, self, target, weight, reduction, ignore_index): ...


def nll_loss_nd(g, self, target, weight, reduction, ignore_index): ...


def cross_entropy_loss(
    g, self, target, weight, reduction, ignore_index, label_smoothing): ...


def binary_cross_entropy_with_logits(
    g, input, target, weight, pos_weight, reduction): ...


def celu(g, self, alpha): ...


def argmax(g, input, dim, keepdim): ...


def argmin(g, input, dim, keepdim): ...


def pow(g, self, exponent): ...


def ge(g, input, other): ...


def le(g, input, other): ...


def unfold(g, input, dimension, size, step): ...


def tensordot(
    g, input_a, input_b, dims_a, dims_b, out: Incomplete | None = ...): ...
