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


        symbolic_registry as symbolic_registry


def register_quantized_ops(domain: str, version: int): ...


def nchw2nhwc(g, input): ...


def nhwc2nchw(g, input): ...


def linear_prepack(g, weight, bias): ...


def linear(g, input, weight, bias, scale, zero_point): ...


def conv_prepack(
    g, input, weight, bias, stride, padding, dilation, groups): ...


def conv2d(
    g, input, weight, bias, stride, padding, dilation, groups, scale,
    zero_point): ...


def conv2d_relu(
    g, input, weight, bias, stride, padding, dilation, groups, scale,
    zero_point): ...


def add(g, input_a, input_b, scale, zero_point): ...


def relu(g, input): ...


def quantize_per_tensor(g, input, scale, zero_point, dtype): ...


def dequantize(g, input): ...


def upsample_nearest2d(
    g, input, output_size, align_corners: Incomplete | None = ...,
    scales_h: Incomplete | None = ..., scales_w: Incomplete | None = ...): ...


def max_pool2d(
    g, input, kernel_size, stride, padding, dilation, ceil_mode): ...


def avg_pool2d(
    g, input, kernel_size, stride, padding, ceil_mode, count_include_pad,
    divisor_override: Incomplete | None = ...): ...


def reshape(g, input, shape): ...


def slice(g, input, dim, start, end, step): ...


def cat(
    g, tensor_list, dim, scale: Incomplete | None = ...,
    zero_point: Incomplete | None = ...): ...


def sigmoid(g, input): ...
