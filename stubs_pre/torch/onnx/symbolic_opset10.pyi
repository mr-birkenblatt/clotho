# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch import _C
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx._globals import GLOBALS as GLOBALS


def div(g, self, other, *args): ...


def sort(g, self, dim, decending, out: Incomplete | None = ...): ...


def topk(g, self, k, dim, largest, sorted, out: Incomplete | None = ...): ...


max_pool1d: Incomplete
max_pool2d: Incomplete
max_pool3d: Incomplete
max_pool1d_with_indices: Incomplete
max_pool2d_with_indices: Incomplete
max_pool3d_with_indices: Incomplete
avg_pool1d: Incomplete
avg_pool2d: Incomplete
avg_pool3d: Incomplete
upsample_nearest1d: Incomplete
upsample_nearest2d: Incomplete
upsample_nearest3d: Incomplete
upsample_linear1d: Incomplete
upsample_bilinear2d: Incomplete
upsample_trilinear3d: Incomplete


def slice(g, self, *args): ...


def flip(g, input, dims): ...


def fmod(g, input, other): ...


def embedding_bag(
    g, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse,
    per_sample_weights, include_last_offset, padding_idx): ...


def fake_quantize_per_tensor_affine(
    g, inputs, scale, zero_point, quant_min: int = ...,
    quant_max: int = ...): ...


def isinf(g, input): ...


def isfinite(g, input): ...


def quantize_per_tensor(g, input, scale, zero_point, dtype): ...


def dequantize(g, input): ...


def nan_to_num(g, input, nan, posinf, neginf): ...


class Quantized:
    domain: str
    @staticmethod
    def linear(g, q_input, q_weight, bias, op_scale, op_zero_point): ...
    @staticmethod
    def add(g, x, y, op_scale, op_zero_point): ...
    @staticmethod
    def add_relu(g, x, y, op_scale, op_zero_point): ...
    @staticmethod
    def mul(g, x, y, op_scale, op_zero_point): ...
    @staticmethod
    def hardswish(g, x, op_scale, op_zero_point): ...

    @staticmethod
    def conv2d_relu(
        g, q_input, q_weight, bias, stride, padding, dilation, groups,
        op_scale, op_zero_point): ...

    @staticmethod
    def conv2d(
        g, q_input, q_weight, bias, stride, padding, dilation, groups,
        op_scale, op_zero_point): ...

    @staticmethod
    def cat(
        g, q_inputs: _C.Value, dim: int, op_scale: _C.Value,
        op_zero_point: _C.Value) -> _C.Value: ...
