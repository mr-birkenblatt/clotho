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


def softmax(g, input, dim, dtype: Incomplete | None = ...): ...


def log_softmax(g, input, dim, dtype: Incomplete | None = ...): ...


def frobenius_norm(
    g, self, dim: Incomplete | None = ..., keepdim: bool = ...): ...


def split(
    g, self, split_size_or_sizes, dim, _outputs: Incomplete | None = ...): ...


def split_with_sizes(
    g, self, split_sizes, dim, _outputs: Incomplete | None = ...): ...


def unsafe_split(
    g, self, split_size_or_sizes, dim, _outputs: Incomplete | None = ...): ...


def unsafe_split_with_sizes(
    g, self, split_sizes, dim, _outputs: Incomplete | None = ...): ...


def unbind(g, self, dim: int = ..., _outputs: Incomplete | None = ...): ...


def nonzero_numpy(g, input, _outputs: Incomplete | None = ...): ...


def where(
    g, condition, self: Incomplete | None = ...,
    other: Incomplete | None = ..., _outputs: Incomplete | None = ...): ...


def fake_quantize_per_channel_affine(
    g, inputs, scale, zero_point, axis, quant_min: int = ...,
    quant_max: int = ...): ...


def fake_quantize_per_tensor_affine(
    g, inputs, scale, zero_point, quant_min: int = ...,
    quant_max: int = ...): ...


sum: Incomplete


def unsafe_chunk(g, self, chunks, dim, _outputs: Incomplete | None = ...): ...


def repeat_interleave(
    g, self, repeats, dim: Incomplete | None = ...,
    output_size: Incomplete | None = ...): ...


def diagonal(g, self, offset, dim1, dim2): ...


class Quantized:
    domain: str
    @staticmethod
    def linear(g, q_input, q_weight, bias, op_scale, op_zero_point): ...

    @staticmethod
    def conv2d(
        g, q_input, q_weight, bias, stride, padding, dilation, groups,
        op_scale, op_zero_point): ...

    @staticmethod
    def conv2d_relu(
        g, q_input, q_weight, bias, stride, padding, dilation, groups,
        op_scale, op_zero_point): ...
