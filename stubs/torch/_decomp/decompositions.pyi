# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,consider-using-from-import
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import torch._prims.utils as utils
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._decomp import register_decomposition as register_decomposition
from torch._prims.wrappers import out_wrapper_multi as out_wrapper_multi
from torch.utils._pytree import tree_flatten as tree_flatten
from torch.utils._pytree import tree_map as tree_map


aten: Incomplete


class Reduction(Enum):
    NONE: int
    MEAN: int
    SUM: int


def type_casts(
    f: Callable, type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND): ...


pw_cast_for_opmath: Incomplete
reduction_complex_to_real: Incomplete
pw_cast_for_int_to_real: Incomplete


def tanh_backward(out_grad: Tensor, y: Tensor): ...


def sigmoid_backward(out_grad: Tensor, y: Tensor): ...


def softplus_backward(
    out_grad: Tensor, x: Tensor, beta: float, threshold: float): ...


def elu(
    self, alpha: float = ..., scale: float = ...,
    input_scale: float = ...) -> Tensor: ...


def elu_backward(
    grad_output: Tensor, alpha: float, scale: float, input_scale: float,
    is_result: bool, self_or_result: Tensor): ...


def hardsigmoid(self) -> Tensor: ...


def hardsigmoid_backward(grad_output: Tensor, self: Tensor): ...


def hardtanh(self, min_val: float = ..., max_val: float = ...) -> Tensor: ...


def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float): ...


def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: float): ...


def hardswish(self) -> Tensor: ...


def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor: ...


def threshold_backward(
    grad_output: Tensor, self: Tensor, threshold: float): ...


def leaky_relu(self, negative_slope: float = ...) -> Tensor: ...


def leaky_relu_backward(
    grad_output: Tensor, self: Tensor,
    negative_slope: float, self_is_result: bool): ...


def gelu(self, approximate: str = ...) -> Tensor: ...


def gelu_backward(grad: Tensor, self: Tensor, approximate: str = ...): ...


def mish_backward(grad_output: Tensor, input: Tensor): ...


def silu(self) -> Tensor: ...


def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor: ...


def softshrink_backward(
    grad_output: Tensor, self: Tensor, lambd: float) -> Tensor: ...


def prelu_backward(
    grad_output: Tensor,
    self: Tensor,
    weight: Tensor) -> Tuple[Tensor, Tensor]: ...


def rrelu_with_noise_backward(
    grad_output: Tensor, self: Tensor, noise: Tensor, lower: float,
    upper: float, training: bool, self_is_result: bool) -> Tensor: ...


def log_sigmoid_backward(
    grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor: ...


def apply_loss_reduction(loss: Tensor, reduction: int): ...


def to_real_dtype(dtype: torch.dtype): ...


def l1_loss(self, target: Tensor, reduction: int = ...) -> Tensor: ...


def l1_loss_backward(
    grad_output: Tensor, self: Tensor,
    target: Tensor, reduction: int = ...): ...


def mse_loss(self, target: Tensor, reduction: int = ...) -> Tensor: ...


def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int): ...


def huber_loss(
    self, target: Tensor, reduction: int = ...,
    delta: float = ...) -> Tensor: ...


def huber_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor,
    reduction: int, delta: float): ...


def nll_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor,
    weight: Optional[Tensor], reduction: int, ignore_index: int,
    total_weight: Tensor) -> Tensor: ...


def nll_loss2d_backward(
    grad_output: Tensor, self: Tensor, target: Tensor,
    weight: Optional[Tensor], reduction: int, ignore_index: int,
    total_weight: Tensor) -> Tensor: ...


def binary_cross_entropy(
    self, target: Tensor, weight: Optional[Tensor] = ...,
    reduction: int = ...) -> Tensor: ...


def binary_cross_entropy_backward(
    grad_output: Tensor, self: Tensor, target: Tensor,
    weight: Optional[Tensor] = ..., reduction: int = ...) -> Tensor: ...


def slice_backward(
    grad_output: Tensor, input_sizes: List[int], dim: int,
    start: int, end: int, step: int): ...


def select_backward(
    grad_output: Tensor, input_sizes: List[int], dim: int, index: int): ...


def diagonal_backward(
    grad_output: Tensor, input_sizes: List[int], offset: int,
    dim1: int, dim2: int): ...


def im2col_backward(
    grad_output: Tensor, input_size: List[int], kernel_size: List[int],
    dilation: List[int], padding: List[int], stride: List[int]) -> Tensor: ...


def col2im_backward(
    grad_output: Tensor, kernel_size: List[int], dilation: List[int],
    padding: List[int], stride: List[int]) -> Tensor: ...


def masked_fill_Scalar(self, mask: Tensor, value: float) -> Tensor: ...


def masked_fill_Tensor(self, mask: Tensor, value: Tensor) -> Tensor: ...


def native_dropout_backward(
    grad_output: Tensor, mask: Tensor, scale: float): ...


def logit(self, eps: Optional[float] = ...) -> Tensor: ...


def logit_backward(
    grad_output: Tensor, self: Tensor,
    eps: Optional[float] = ...) -> Tensor: ...


def native_dropout(input: Tensor, p: float, train: Optional[bool]): ...


def addcdiv(self, tensor1: Tensor, tensor2: Tensor, value: float = ...): ...


def addcmul(self, tensor1: Tensor, tensor2: Tensor, value: float = ...): ...


def rsub_Tensor(self, other: Tensor, alpha: float = ...) -> Tensor: ...


def rsub_Scalar(self, other: float, alpha: float = ...) -> Tensor: ...


def embedding(
    weight: Tensor, indices: Tensor, padding_idx: int = ...,
    scale_grad_by_freq: bool = ..., sparse: bool = ...) -> Tensor: ...


def embedding_dense_backward(
    grad_output: Tensor, indices: Tensor, num_weights: int,
    padding_idx: int, scale_grad_by_freq: bool): ...


def prod(x: List[int]): ...


def split_with_sizes(
    self, split_sizes: List[int], dim: int = ...) -> List[Tensor]: ...


def split(self, split_size: int, dim: int = ...) -> List[Tensor]: ...


def addmm(
    self, mat1: Tensor, mat2: Tensor, beta: int = ..., alpha: int = ...): ...


def native_layer_norm(
    input: Tensor, normalized_shape: List[int], weight: Optional[Tensor],
    bias: Optional[Tensor], eps: float) -> Tuple[Tensor, Tensor, Tensor]: ...


def native_layer_norm_backward(
    grad_out: Tensor, input: Tensor, normalized_shape: List[int],
    mean: Tensor, rstd: Tensor, weight: Optional[Tensor],
    bias: Optional[Tensor], output_mask: List[bool],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]: ...


def native_batch_norm(
    input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
    running_mean: Optional[Tensor], running_var: Optional[Tensor],
    training: bool, momentum: float, eps: float,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...


def clamp_min(self, min: float): ...


def clamp_max(self, max: float): ...


def logical_xor(self, other: Tensor) -> Tensor: ...


def logical_not(self) -> Tensor: ...


def xlogy(self, other: Tensor) -> Tensor: ...


def var_correction(
    x: Tensor, dims: Optional[List[int]], correction: Optional[int] = ...,
    keepdim: bool = ...): ...


def std_decomposition(
    x: Tensor, dims: List[int], correction: int = ...,
    keepdim: bool = ...): ...


def detach_decomposition(x): ...


def cudnn_batch_norm(
    input: Tensor, weight: Tensor, bias: Optional[Tensor],
    running_mean: Optional[Tensor], running_var: Optional[Tensor],
    training: bool, exponential_average_factor: float, epsilon: float): ...


def cudnn_batch_norm_backward(
    input: Tensor, grad_output: Tensor, weight: Tensor,
    running_mean: Optional[Tensor], running_var: Optional[Tensor],
    save_mean: Optional[Tensor], save_var: Optional[Tensor],
    epsilon: float, reserveSpace: Tensor): ...


def rot90(self, k: int = ..., dims: List[int] = ...) -> Tensor: ...


def transpose_int(self, dim0: int, dim1: int) -> Tensor: ...


def t(self) -> Tensor: ...


def check_stack_inputs(tensors: List[Tensor]): ...


def get_stack_inputs(tensors: List[Tensor], dim: int): ...


def stack(tensors: List[Tensor], dim: int = ...) -> Tensor: ...


def logsumexp(self, dim: List[int], keepdim: bool = ...) -> Tensor: ...


def trace(self) -> Tensor: ...


def log_sigmoid_forward(self) -> Tuple[Tensor, Tensor]: ...
