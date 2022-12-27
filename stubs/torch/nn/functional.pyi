# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable, List, Optional, Tuple, Union

from _typeshed import Incomplete
from torch._torch_docs import reproducibility_notes as reproducibility_notes
from torch._torch_docs import tf32_notes as tf32_notes
from torch.types import _dtype as DType

from .._jit_internal import boolean_dispatch as boolean_dispatch
from .._jit_internal import BroadcastingList1 as BroadcastingList1
from .._jit_internal import BroadcastingList2 as BroadcastingList2
from .._jit_internal import BroadcastingList3 as BroadcastingList3
from ..overrides import handle_torch_function as handle_torch_function
from ..overrides import has_torch_function as has_torch_function
from ..overrides import has_torch_function_unary as has_torch_function_unary
from ..overrides import (
    has_torch_function_variadic as has_torch_function_variadic,
)
from . import grad as grad
from .modules import utils as utils


Tensor: Incomplete
conv1d: Incomplete
conv2d: Incomplete
conv3d: Incomplete
conv_transpose1d: Incomplete
conv_transpose2d: Incomplete
conv_transpose3d: Incomplete
conv_tbc: Incomplete
avg_pool1d: Incomplete
avg_pool2d: Incomplete
avg_pool3d: Incomplete


def fractional_max_pool2d_with_indices(
    input: Tensor, kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = ...,
    output_ratio: Optional[BroadcastingList2[float]] = ...,
    return_indices: bool = ..., _random_samples: Optional[
            Tensor] = ...) -> Tuple[Tensor, Tensor]: ...


fractional_max_pool2d: Incomplete


def fractional_max_pool3d_with_indices(
    input: Tensor, kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = ...,
    output_ratio: Optional[BroadcastingList3[float]] = ...,
    return_indices: bool = ..., _random_samples: Optional[
            Tensor] = ...) -> Tuple[Tensor, Tensor]: ...


fractional_max_pool3d: Incomplete


def max_pool1d_with_indices(
    input: Tensor, kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = ...,
    padding: BroadcastingList1[int] = ...,
    dilation: BroadcastingList1[int] = ..., ceil_mode: bool = ...,
    return_indices: bool = ...) -> Tuple[Tensor, Tensor]: ...


max_pool1d: Incomplete


def max_pool2d_with_indices(
    input: Tensor, kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = ...,
    padding: BroadcastingList2[int] = ...,
    dilation: BroadcastingList2[int] = ..., ceil_mode: bool = ...,
    return_indices: bool = ...) -> Tuple[Tensor, Tensor]: ...


max_pool2d: Incomplete


def max_pool3d_with_indices(
    input: Tensor, kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = ...,
    padding: BroadcastingList3[int] = ...,
    dilation: BroadcastingList3[int] = ..., ceil_mode: bool = ...,
    return_indices: bool = ...) -> Tuple[Tensor, Tensor]: ...


max_pool3d: Incomplete


def max_unpool1d(
    input: Tensor, indices: Tensor, kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = ...,
    padding: BroadcastingList1[int] = ...,
    output_size: Optional[BroadcastingList1[int]] = ...) -> Tensor: ...


def max_unpool2d(
    input: Tensor, indices: Tensor, kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = ...,
    padding: BroadcastingList2[int] = ...,
    output_size: Optional[BroadcastingList2[int]] = ...) -> Tensor: ...


def max_unpool3d(
    input: Tensor, indices: Tensor, kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = ...,
    padding: BroadcastingList3[int] = ...,
    output_size: Optional[BroadcastingList3[int]] = ...) -> Tensor: ...


def lp_pool2d(
    input: Tensor, norm_type: Union[int, float],
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[
                    int]] = ..., ceil_mode: bool = ...) -> Tensor: ...


def lp_pool1d(
    input: Tensor, norm_type: Union[int, float], kernel_size: int,
    stride: Optional[BroadcastingList1[
                    int]] = ..., ceil_mode: bool = ...) -> Tensor: ...


def adaptive_max_pool1d_with_indices(
    input: Tensor, output_size: BroadcastingList1[int],
    return_indices: bool = ...) -> Tuple[Tensor, Tensor]: ...


adaptive_max_pool1d: Incomplete


def adaptive_max_pool2d_with_indices(
    input: Tensor, output_size: BroadcastingList2[int],
    return_indices: bool = ...) -> Tuple[Tensor, Tensor]: ...


adaptive_max_pool2d: Incomplete


def adaptive_max_pool3d_with_indices(
    input: Tensor, output_size: BroadcastingList3[int],
    return_indices: bool = ...) -> Tuple[Tensor, Tensor]: ...


adaptive_max_pool3d: Incomplete
adaptive_avg_pool1d: Incomplete


def adaptive_avg_pool2d(
    input: Tensor, output_size: BroadcastingList2[int]) -> Tensor: ...


def adaptive_avg_pool3d(
    input: Tensor, output_size: BroadcastingList3[int]) -> Tensor: ...


def dropout(
    input: Tensor, p: float = ..., training: bool = ...,
    inplace: bool = ...) -> Tensor: ...


def alpha_dropout(
    input: Tensor, p: float = ..., training: bool = ...,
    inplace: bool = ...) -> Tensor: ...


def dropout1d(
    input: Tensor, p: float = ..., training: bool = ...,
    inplace: bool = ...) -> Tensor: ...


def dropout2d(
    input: Tensor, p: float = ..., training: bool = ...,
    inplace: bool = ...) -> Tensor: ...


def dropout3d(
    input: Tensor, p: float = ..., training: bool = ...,
    inplace: bool = ...) -> Tensor: ...


def feature_alpha_dropout(
    input: Tensor, p: float = ..., training: bool = ...,
    inplace: bool = ...) -> Tensor: ...


threshold: Incomplete
threshold_: Incomplete


def relu(input: Tensor, inplace: bool = ...) -> Tensor: ...


relu_: Incomplete


def glu(input: Tensor, dim: int = ...) -> Tensor: ...


def hardtanh(
    input: Tensor, min_val: float = ..., max_val: float = ...,
    inplace: bool = ...) -> Tensor: ...


hardtanh_: Incomplete


def relu6(input: Tensor, inplace: bool = ...) -> Tensor: ...


def elu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...


elu_: Incomplete


def selu(input: Tensor, inplace: bool = ...) -> Tensor: ...


selu_: Incomplete


def celu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...


celu_: Incomplete


def leaky_relu(
    input: Tensor, negative_slope: float = ...,
    inplace: bool = ...) -> Tensor: ...


leaky_relu_: Incomplete
prelu: Incomplete


def rrelu(
    input: Tensor, lower: float = ..., upper: float = ...,
    training: bool = ..., inplace: bool = ...) -> Tensor: ...


rrelu_: Incomplete
logsigmoid: Incomplete
gelu: Incomplete
hardshrink: Incomplete


def tanhshrink(input): ...


def softsign(input): ...


softplus: Incomplete


def softmin(
    input: Tensor, dim: Optional[int] = ..., _stacklevel: int = ...,
    dtype: Optional[DType] = ...) -> Tensor: ...


def softmax(
    input: Tensor, dim: Optional[int] = ..., _stacklevel: int = ...,
    dtype: Optional[DType] = ...) -> Tensor: ...


def gumbel_softmax(
    logits: Tensor, tau: float = ..., hard: bool = ..., eps: float = ...,
    dim: int = ...) -> Tensor: ...


def log_softmax(
    input: Tensor, dim: Optional[int] = ..., _stacklevel: int = ...,
    dtype: Optional[DType] = ...) -> Tensor: ...


softshrink: Incomplete


def tanh(input): ...


def sigmoid(input): ...


def hardsigmoid(input: Tensor, inplace: bool = ...) -> Tensor: ...


linear: Incomplete
bilinear: Incomplete


def silu(input: Tensor, inplace: bool = ...) -> Tensor: ...


def mish(input: Tensor, inplace: bool = ...) -> Tensor: ...


def hardswish(input: Tensor, inplace: bool = ...) -> Tensor: ...


def embedding(
    input: Tensor, weight: Tensor, padding_idx: Optional[int] = ...,
    max_norm: Optional[float] = ..., norm_type: float = ...,
    scale_grad_by_freq: bool = ..., sparse: bool = ...) -> Tensor: ...


def embedding_bag(
    input: Tensor, weight: Tensor, offsets: Optional[Tensor] = ...,
    max_norm: Optional[float] = ..., norm_type: float = ...,
    scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ...,
    per_sample_weights: Optional[Tensor] = ...,
    include_last_offset: bool = ..., padding_idx: Optional[
            int] = ...) -> Tensor: ...


def batch_norm(
    input: Tensor, running_mean: Optional[Tensor],
    running_var: Optional[Tensor], weight: Optional[Tensor] = ...,
    bias: Optional[Tensor] = ..., training: bool = ...,
    momentum: float = ..., eps: float = ...) -> Tensor: ...


def instance_norm(
    input: Tensor, running_mean: Optional[Tensor] = ...,
    running_var: Optional[Tensor] = ..., weight: Optional[Tensor] = ...,
    bias: Optional[Tensor] = ..., use_input_stats: bool = ...,
    momentum: float = ..., eps: float = ...) -> Tensor: ...


def layer_norm(
    input: Tensor, normalized_shape: List[int],
    weight: Optional[Tensor] = ..., bias: Optional[Tensor] = ...,
    eps: float = ...) -> Tensor: ...


def group_norm(
    input: Tensor, num_groups: int, weight: Optional[Tensor] = ...,
    bias: Optional[Tensor] = ..., eps: float = ...) -> Tensor: ...


def local_response_norm(
    input: Tensor, size: int, alpha: float = ..., beta: float = ...,
    k: float = ...) -> Tensor: ...


def ctc_loss(
    log_probs: Tensor, targets: Tensor, input_lengths: Tensor,
    target_lengths: Tensor, blank: int = ..., reduction: str = ...,
    zero_infinity: bool = ...) -> Tensor: ...


def nll_loss(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
    size_average: Optional[bool] = ..., ignore_index: int = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def poisson_nll_loss(
    input: Tensor, target: Tensor, log_input: bool = ..., full: bool = ...,
    size_average: Optional[bool] = ..., eps: float = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def gaussian_nll_loss(
    input: Tensor, target: Tensor, var: Tensor, full: bool = ...,
    eps: float = ..., reduction: str = ...) -> Tensor: ...


def kl_div(
    input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ..., reduction: str = ...,
    log_target: bool = ...) -> Tensor: ...


def cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
    size_average: Optional[bool] = ..., ignore_index: int = ...,
    reduce: Optional[bool] = ..., reduction: str = ...,
    label_smoothing: float = ...) -> Tensor: ...


def binary_cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ...) -> Tensor: ...


def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ..., pos_weight: Optional[Tensor] = ...) -> Tensor: ...


def smooth_l1_loss(
    input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
    reduce: Optional[
            bool] = ..., reduction: str = ...,
    beta: float = ...) -> Tensor: ...


def huber_loss(
    input: Tensor, target: Tensor, reduction: str = ...,
    delta: float = ...) -> Tensor: ...


def l1_loss(
    input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def mse_loss(
    input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def margin_ranking_loss(
    input1: Tensor, input2: Tensor, target: Tensor, margin: float = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ...) -> Tensor: ...


def hinge_embedding_loss(
    input: Tensor, target: Tensor, margin: float = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ...) -> Tensor: ...


def multilabel_margin_loss(
    input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def soft_margin_loss(
    input: Tensor, target: Tensor, size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


def multilabel_soft_margin_loss(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ...) -> Tensor: ...


def cosine_embedding_loss(
    input1: Tensor, input2: Tensor, target: Tensor, margin: float = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ...) -> Tensor: ...


def multi_margin_loss(
    input: Tensor, target: Tensor, p: int = ..., margin: float = ...,
    weight: Optional[Tensor] = ..., size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ..., reduction: str = ...) -> Tensor: ...


pixel_shuffle: Incomplete
pixel_unshuffle: Incomplete
channel_shuffle: Incomplete
native_channel_shuffle: Incomplete
GRID_SAMPLE_INTERPOLATION_MODES: Incomplete
GRID_SAMPLE_PADDING_MODES: Incomplete


def grid_sample(
    input: Tensor, grid: Tensor, mode: str = ..., padding_mode: str = ...,
    align_corners: Optional[bool] = ...) -> Tensor: ...


def affine_grid(
    theta: Tensor, size: List[int], align_corners: Optional[
            bool] = ...) -> Tensor: ...


pad: Incomplete
pairwise_distance: Incomplete
pdist: Incomplete
cosine_similarity: Incomplete
one_hot: Incomplete


def triplet_margin_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = ...,
    p: float = ..., eps: float = ..., swap: bool = ...,
    size_average: Optional[bool] = ..., reduce: Optional[bool] = ...,
    reduction: str = ...) -> Tensor: ...


def triplet_margin_with_distance_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor, *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = ...,
    margin: float = ..., swap: bool = ..., reduction: str = ...) -> Tensor: ...


def normalize(
    input: Tensor, p: float = ..., dim: int = ..., eps: float = ...,
    out: Optional[Tensor] = ...) -> Tensor: ...


def assert_int_or_pair(
    arg: List[int], arg_name: str, message: str) -> None: ...


def unfold(
    input: Tensor, kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = ...,
    padding: BroadcastingList2[int] = ...,
    stride: BroadcastingList2[int] = ...) -> Tensor: ...


def fold(
    input: Tensor, output_size: BroadcastingList2[int],
    kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = ...,
    padding: BroadcastingList2[int] = ...,
    stride: BroadcastingList2[int] = ...) -> Tensor: ...


def multi_head_attention_forward(
    query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int,
    num_heads: int, in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor], bias_k: Optional[Tensor],
    bias_v: Optional[Tensor], add_zero_attn: bool, dropout_p: float,
    out_proj_weight: Tensor, out_proj_bias: Optional[Tensor],
    training: bool = ..., key_padding_mask: Optional[Tensor] = ...,
    need_weights: bool = ..., attn_mask: Optional[Tensor] = ...,
    use_separate_proj_weight: bool = ...,
    q_proj_weight: Optional[Tensor] = ...,
    k_proj_weight: Optional[Tensor] = ...,
    v_proj_weight: Optional[Tensor] = ..., static_k: Optional[Tensor] = ...,
    static_v: Optional[
            Tensor] = ..., average_attn_weights: bool = ...) -> Tuple[
        Tensor, Optional[Tensor]]: ...
