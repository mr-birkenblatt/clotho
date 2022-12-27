# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional

import torch.onnx
from _typeshed import Incomplete
from torch import _C
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx._globals import GLOBALS as GLOBALS


def unused(g): ...


def reshape(g, self, shape): ...


def reshape_as(g, self, other): ...


def add(g, self, other, alpha: Incomplete | None = ...): ...


def sub(g, self, other, alpha: Incomplete | None = ...): ...


def rsub(g, self, other, alpha: Incomplete | None = ...): ...


def mul(g, self, other): ...


def div(g, self, other, *args): ...


def addcmul(g, self, tensor1, tensor2, value: float = ...): ...


def floor_divide(g, self, other): ...


def floordiv(g, self, other): ...


def true_divide(g, self, other): ...


def reciprocal(g, self): ...


def cat(g, tensor_list, dim): ...


def stack(g, tensor_list, dim): ...


def mm(g, self, other): ...


def bmm(g, self, other): ...


def matmul(g, self, other): ...


def addmm(g, self, mat1, mat2, beta, alpha): ...


def neg(g, self): ...


def sqrt(g, self): ...


def rsqrt(g, self): ...


def tanh(g, self): ...


def sin(g, self): ...


def cos(g, self): ...


def tan(g, self): ...


def asin(g, self): ...


def acos(g, self): ...


def atan(g, self): ...


def sigmoid(g, self): ...


def sign(g, self): ...


def overload_by_arg_count(fn): ...


sum: Incomplete
mean: Incomplete
prod: Incomplete


def cumsum(g, input, dim, dtype): ...


def t(g, self): ...


def expand(g, self, size, implicit): ...


def expand_as(g, self, other): ...


def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse): ...


def embedding_bag(
    g, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse,
    per_sample_weights, include_last_offset, padding_idx): ...


def size(g, self, dim: Incomplete | None = ...): ...


def transpose(g, self, dim0, dim1): ...


def permute(g, self, dims): ...


def view(g, self, size): ...


def view_as(g, self, other): ...


def unsafe_chunk(g, self, chunks, dim, _outputs: Incomplete | None = ...): ...


def split(
    g, self, split_size_or_sizes, dim, _outputs: Incomplete | None = ...): ...


def unsafe_split(
    g, self, split_size_or_sizes, dim, _outputs: Incomplete | None = ...): ...


def split_with_sizes(
    g, self, split_sizes, dim, _outputs: Incomplete | None = ...): ...


def unsafe_split_with_sizes(
    g, self, split_sizes, dim, _outputs: Incomplete | None = ...): ...


def unbind(g, self, dim: int = ..., _outputs: Incomplete | None = ...): ...


def select(g, self, dim, index): ...


def square(g, self): ...


def squeeze(g, self, dim: Incomplete | None = ...): ...


def prelu(g, self, weight): ...


def silu(g, input): ...


def mish(g, input): ...


def op_with_optional_float_cast(g, op_name, *args, **kwargs): ...


def relu(g, input): ...


def relu6(g, input): ...


def ceil(g, input): ...


def floor(g, input): ...


def threshold(g, self, threshold, value): ...


def leaky_relu(g, input, negative_slope, inplace: bool = ...): ...


def glu(g, input, dim): ...


def softmax(g, input, dim, dtype: Incomplete | None = ...): ...


def softplus(g, self, beta, threshold): ...


def get_pool_ceil_padding(input, kernel_size, stride, padding): ...


max_pool1d: Incomplete
max_pool2d: Incomplete
max_pool3d: Incomplete
max_pool1d_with_indices: Incomplete
max_pool2d_with_indices: Incomplete
max_pool3d_with_indices: Incomplete
avg_pool1d: Incomplete
avg_pool2d: Incomplete
avg_pool3d: Incomplete
adaptive_avg_pool1d: Incomplete
adaptive_avg_pool2d: Incomplete
adaptive_avg_pool3d: Incomplete
adaptive_max_pool1d: Incomplete
adaptive_max_pool2d: Incomplete
adaptive_max_pool3d: Incomplete


def constant_pad_nd(g, input, padding, value): ...


def reflection_pad(g, input, padding): ...


def replication_pad(g, input, padding): ...


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def pad(g, input, pad, mode, value): ...


upsample_nearest1d: Incomplete
upsample_nearest2d: Incomplete
upsample_nearest3d: Incomplete
upsample_linear1d: Incomplete
upsample_bilinear2d: Incomplete
upsample_trilinear3d: Incomplete


def bitwise_not(g, inp): ...


def wrap_logical_op_with_cast_to(to_type): ...


def wrap_logical_op_with_cast_to_and_from(to_type): ...


def wrap_logical_op_with_negation(func): ...


def eq(g, self, other): ...


def ne(g, self, other): ...


def gt(g, input, other): ...


def gt_impl(g, input, other): ...


def lt(g, input, other): ...


def lt_impl(g, input, other): ...


def ge(g, input, other): ...


def le(g, input, other): ...


def logical_and(g, input, other): ...


def logical_or(g, input, other): ...


def logical_xor(g, input, other): ...


def where(
    g, condition, self: Incomplete | None = ...,
    other: Incomplete | None = ..., _outputs: Incomplete | None = ...): ...


def log_softmax(g, input, dim, dtype: Incomplete | None = ...): ...


def conv1d(g, input, weight, bias, stride, padding, dilation, groups): ...


def conv2d(g, input, weight, bias, stride, padding, dilation, groups): ...


def conv3d(g, input, weight, bias, stride, padding, dilation, groups): ...


def conv_transpose1d(
    g, input, weight, bias, stride, padding, output_padding, groups,
    dilation): ...


def conv_transpose2d(
    g, input, weight, bias, stride, padding, output_padding, groups,
    dilation): ...


def conv_transpose3d(
    g, input, weight, bias, stride, padding, output_padding, groups,
    dilation): ...


def batch_norm(
    g, input, weight, bias, running_mean, running_var, training, momentum,
    eps, cudnn_enabled): ...


def layer_norm(
    g, input, normalized_shape, weight, bias, eps, cudnn_enable): ...


def instance_norm(
    g, input, weight, bias, running_mean, running_var, use_input_stats,
    momentum, eps, cudnn_enabled): ...


def unfold(g, input, dimension, size, step): ...


def elu(g, input, alpha, scale, input_scale): ...


def selu(g, input): ...


def index_select(g, self, dim, index): ...


def index_put(g, self, indices_list_value, values, accumulate): ...


def index_fill(g, self, dim, index, value): ...


def index_copy(g, self, dim, index, source): ...


def bucketize(
    g, self, boundaries, out_int32: bool = ..., right: bool = ...): ...


def type_as(g, self, other): ...


def cosine_similarity(g, x1, x2, dim, eps): ...


def pairwise_distance(g, input1, input2, p, eps, keepdim): ...


def clone(g, input, unused_memory_format): ...


def abs(g, self): ...


def log(g, self): ...


def log1p(g, self): ...


def log10(g, self): ...


def pow(g, self, exponent): ...


def clamp(g, self, min, max): ...


def clamp_min(g, self, min): ...


def clamp_max(g, self, max): ...


def max(
    g, self, dim_or_y: Incomplete | None = ...,
    keepdim: Incomplete | None = ...): ...


def maximum(g, input, other): ...


def min(
    g, self, dim_or_y: Incomplete | None = ...,
    keepdim: Incomplete | None = ...): ...


def minimum(g, input, other): ...


def amax(g, self, dim, keepdim): ...


def amin(g, self, dim, keepdim): ...


def aminmax(g, self, dim, keepdim): ...


def exp(g, self): ...


def dropout(g, input, p, train): ...


feature_dropout: Incomplete
alpha_dropout: Incomplete
feature_alpha_dropout: Incomplete
dropout_ = dropout
feature_dropout_ = feature_dropout
alpha_dropout_ = alpha_dropout
feature_alpha_dropout_ = feature_alpha_dropout


def norm(g, self, p, dim, keepdim): ...


def conv_tbc(g, input, weight, bias, pad): ...


name: Incomplete


def empty(
    g, sizes, dtype, layout, device, pin_memory: bool = ...,
    memory_format: Incomplete | None = ...): ...


def empty_like(
    g, input, dtype: Incomplete | None = ...,
    layout: Incomplete | None = ..., device: Incomplete | None = ...,
    pin_memory: bool = ..., memory_format: Incomplete | None = ...): ...


def new_empty(
    g, self, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def scalar_tensor(g, scalar, dtype, *options): ...


def tensor(
    g, data, dtype: Incomplete | None = ..., device: Incomplete | None = ...,
    requires_grad: bool = ...): ...


def as_tensor(
    g, data, dtype: Incomplete | None = ...,
    device: Incomplete | None = ...): ...


def zeros(g, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def zeros_like(
    g, input, dtype: Incomplete | None = ...,
    layout: Incomplete | None = ..., device: Incomplete | None = ...,
    pin_memory: bool = ..., memory_format: Incomplete | None = ...): ...


def new_zeros(
    g, self, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def ones(g, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def ones_like(
    g, input, dtype: Incomplete | None = ...,
    layout: Incomplete | None = ..., device: Incomplete | None = ...,
    pin_memory: bool = ..., memory_format: Incomplete | None = ...): ...


def new_ones(
    g, self, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def full(g, sizes, value, dtype, layout, device, pin_memory: bool = ...): ...


def full_like(
    g, input, fill_value, dtype: Incomplete | None = ...,
    layout: Incomplete | None = ..., device: Incomplete | None = ...,
    pin_memory: bool = ..., memory_format: Incomplete | None = ...): ...


def new_full(
    g, self, size, fill_value, dtype, layout, device,
    pin_memory: bool = ...): ...


def eye(g, *args): ...


def slice(g, self, *args): ...


def hardtanh(g, self, min_val, max_val): ...


def hardswish(g, self): ...


def hardsigmoid(g, self): ...


def tanhshrink(g, self): ...


def hardshrink(g, self, lambd): ...


def softshrink(g, self, lambd): ...


def alias(g, self): ...


def unsqueeze(g, self, dim): ...


def sort(g, self, dim, decending, out: Incomplete | None = ...): ...


def numel(g, self): ...


def topk(g, self, k, dim, largest, sorted, out: Incomplete | None = ...): ...


def to(g, self, *args): ...


def repeat(g, self, repeats): ...


def repeat_interleave(
    g, self, repeats, dim: Incomplete | None = ...,
    output_size: Incomplete | None = ...): ...


def pixel_shuffle(g, self, upscale_factor): ...


def pixel_unshuffle(g, self, downscale_factor): ...


def lstm(g, *args): ...


def lstm_cell(g, self, hidden, w_ih, w_hh, b_ih, b_hh): ...


gru: Incomplete
rnn_tanh: Incomplete
rnn_relu: Incomplete


def detach(g, input): ...


def contiguous(g, input, memory_format): ...


def randn(g, shapes, dtype, *options): ...


def rand(g, shapes, dtype, *options): ...


def randn_like(
    g, self, dtype, layout: Incomplete | None = ...,
    device: Incomplete | None = ..., pin_memory: bool = ...,
    memory_format: Incomplete | None = ...): ...


def rand_like(
    g, self, dtype, layout: Incomplete | None = ...,
    device: Incomplete | None = ..., pin_memory: bool = ...,
    memory_format: Incomplete | None = ...): ...


def rrelu(g, input, lower, upper, training, generator): ...


def bernoulli(
    g, input, generator: Incomplete | None = ...,
    out: Incomplete | None = ...): ...


def log_sigmoid(g, input): ...


def erf(g, input): ...


def flatten(g, input, start_dim, end_dim): ...


def nonzero(g, input): ...


def nonzero_numpy(g, input, _outputs: Incomplete | None = ...): ...


def isnan(g, input): ...


def narrow(g, input, dim, start, length): ...


def argmax(g, input, dim, keepdim): ...


def argmin(g, input, dim, keepdim): ...


def scatter(g, self, dim, index, src): ...


def scatter_add(g, self, dim, index, src): ...


def log2(g, self): ...


def is_floating_point(g, self): ...


def one_hot(g, self, num_classes): ...


def gather(g, self, dim, index, sparse_grad: bool = ...): ...


def std(g, input, *args): ...


def var(g, input, *args): ...


def var_mean(g, input, *args): ...


def std_mean(g, input, *args): ...


def logsumexp(g, input, dim, keepdim): ...


def arange(g, *args): ...


def linspace(g, start, end, steps, dtype, layout, device, pin_memory): ...


def lift(g, self): ...


def masked_fill(g, self, mask, value): ...


def index(g, self, index): ...


def linalg_norm(g, self, ord, dim, keepdim, dtype): ...


def linalg_vector_norm(g, self, ord, dim, keepdim, dtype): ...


def linalg_matrix_norm(g, self, ord, dim, keepdim, dtype): ...


def linalg_cross(g, input, other, dim: int = ...): ...


def frobenius_norm(
    g, self, dim: Incomplete | None = ..., keepdim: bool = ...): ...


def multinomial(
    g, input, num_samples, replacement: bool = ...,
    generator: Incomplete | None = ...): ...


def baddbmm(g, self, batch1, batch2, beta, alpha): ...


def meshgrid(g, tensor_list, indexing: Optional[str] = ...): ...


def remainder(g, input, other): ...


def gelu(g, self: torch._C.Value, approximate: str = ...): ...


def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled): ...


def dim(g, self): ...


def item(g, self): ...


def take(g, self, index): ...


def kl_div(g, input, target, reduction, log_target): ...


def as_strided(g, self, sizes, strides, offset: Incomplete | None = ...): ...


def linear(g, input, weight, bias): ...


def hann_window(
    g, window_length, periodic: bool = ..., dtype: Incomplete | None = ...,
    layout: Incomplete | None = ..., device: Incomplete | None = ...,
    pin_memory: Incomplete | None = ..., requires_grad: bool = ...): ...


def mv(g, self, vec): ...


def dot(g, self, other): ...


def fill(g, self, value): ...


def index_add(g, self, dim, index, other, alpha: Incomplete | None = ...): ...


def roll(g, self, shifts, dims): ...


def cross(g, input, other, dim: Incomplete | None = ...): ...


def cdist(g, x1, x2, p: float = ..., compute_mode: str = ...): ...


def broadcast_tensors(g, self): ...


class Prim:
    domain: str
    @staticmethod
    def ConstantSplit(g, self, split_size, dim): ...
    @staticmethod
    def ConstantChunk(g, self, chunks, dim): ...
    @staticmethod
    def shape(g, self): ...
    @staticmethod
    def max(g, self, other): ...
    @staticmethod
    def min(g, self, other: Incomplete | None = ...): ...
    @staticmethod
    def data(g, self): ...
    @staticmethod
    def ListConstruct(g, *inputs, **kwargs) -> None: ...
    @staticmethod
    def ListUnpack(g, *inputs, **kwargs) -> Optional[List[_C.Value]]: ...
    @staticmethod
    def TupleConstruct(g, *inputs, **kwargs) -> None: ...
    @staticmethod
    def Uninitialized(g, *inputs, **kwargs) -> None: ...
    @staticmethod
    def unchecked_cast(g, self): ...
    @staticmethod
    def dtype(g, self): ...
    @staticmethod
    def tolist(g, input, dim_val, elem_ty_val): ...
    @staticmethod
    def device(ctx: torch.onnx.SymbolicContext, g, *inputs, **kwargs): ...
    @staticmethod
    def Loop(ctx: torch.onnx.SymbolicContext, g, *inputs, **attrs): ...
    @staticmethod
    def If(ctx: torch.onnx.SymbolicContext, g, *inputs, **attrs): ...
    @staticmethod
    def Constant(ctx: torch.onnx.SymbolicContext, g, *inputs, **attrs): ...


class Onnx:
    domain: str
    @staticmethod
    def Placeholder(ctx: torch.onnx.SymbolicContext, g, *inputs, **attrs): ...
