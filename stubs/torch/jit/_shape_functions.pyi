# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from _typeshed import Incomplete


number = Union[int, float]


def broadcast(a: List[int], b: List[int]): ...


def broadcast_three(a: List[int], b: List[int], c: List[int]): ...


def broadcast_one_three(a: List[int], b: Any, c: List[int]): ...


def adaptive_avg_pool2d(self, out: List[int]): ...


def unary(self): ...


def broadcast_inplace(a: List[int], b: List[int]): ...


def expand(self, sizes: List[int]): ...


def expand_one_unused(self, sizes: List[int], inp0: Any): ...


def infer_size_impl(shape: List[int], numel: int) -> List[int]: ...


def numel(sizes: List[int]): ...


def view(self, sizes: List[int]): ...


def view_one_unused(self, sizes: List[int], *, implicit: bool = ...): ...


def mean_dim(self, dims: List[int], keep_dim: bool, dt: Any): ...


def max_dim(self, dim: int, keep_dim: bool): ...


def div_rtn(x: int, y: int): ...


def pooling_output_shape_pad_lr(
    inputSize: int, kernelSize: int, pad_l: int, pad_r: int, stride: int,
        dilation: int, ceil_mode: bool): ...


def pooling_output_shape(
    inputSize: int, kernelSize: int, pad_l: int, stride: int, dilation: int,
        ceil_mode: bool): ...


def pool2d_shape_check(
    input: List[int], kH: int, kW: int, dH: int, dW: int, padH: int,
        padW: int, dilationH: int, dilationW: int, nInputPlane: int,
        inputHeight: int, inputWidth: int, outputHeight: int,
        outputWidth: int): ...


def max_pool2d(
    input: List[int], kernel_size: List[int], stride: List[int],
        padding: List[int], dilation: List[int], ceil_mode: bool): ...


def max_pool2d_with_indices(
    input: List[int], kernel_size: List[int], stride: List[int],
        padding: List[int], dilation: List[int], ceil_mode: bool): ...


def upsample_nearest2d(
    input: List[int], output_size: Optional[List[int]],
        scale_factors: Optional[List[float]]): ...


def mm(self, mat2: List[int]): ...


def dot(self, tensor: List[int]): ...


def mv(self, vec: List[int]): ...


def unsqueeze(li: List[int], dim: int): ...


def squeeze_nodim(li: List[int]): ...


def squeeze(li: List[int], dim: int): ...


def index_select(self, dim: int, index: List[int]): ...


def embedding(
    weight: List[int], indices: List[int], padding_idx: int = ...,
        scale_grad_by_freq: bool = ..., sparse: bool = ...): ...


def max_int(): ...


def slice(
    self, dim: int, start: Optional[int], end: Optional[int], step: int): ...


def check_cat_no_zero_dim(tensors: List[List[int]]): ...


def legacy_cat_wrap_dim(dim: int, tensor_sizes: List[List[int]]): ...


def should_skip(tensor: List[int]): ...


def check_cat_shape_except_dim(
    first: List[int], second: List[int], dimension: int, index: int): ...


def cat(tensors: List[List[int]], dim: int): ...


def select(self, dim: int, index: int): ...


def matmul(tensor1: List[int], tensor2: List[int]): ...


def t(self): ...


def transpose(self, dim0: int, dim1: int): ...


def linear(input: List[int], weight: List[int], bias: Optional[List[int]]): ...


def addmm(self, mat1: List[int], mat2: List[int], beta: Any, alpha: Any): ...


def check_non_negative(array: List[int]) -> bool: ...


def check_shape_forward(
    input: List[int], weight_sizes: List[int], bias: Optional[List[int]],
        stride: List[int], padding: List[int], dilation: List[int],
        groups: int): ...


def conv_output_size(
    input_size: List[int], weight_size: List[int], bias: Optional[List[int]],
        stride: List[int], padding: List[int], dilation: List[int],
        groups: int): ...


def conv1d(
    input: List[int], weight: List[int], bias: Optional[List[int]],
        stride: List[int], padding: List[int], dilation: List[int],
        groups: int): ...


def conv2d(
    input: List[int], weight: List[int], bias: Optional[List[int]],
        stride: List[int], padding: List[int], dilation: List[int],
        groups: int): ...


def batch_norm(
    input: List[int], weight: Optional[List[int]], bias: Optional[List[int]],
        running_mean: Optional[List[int]], running_var: Optional[List[int]],
        training: bool, momentum: float, eps: float, cudnn_enabled: bool): ...


def conv3d(
    input: List[int], weight: List[int], bias: Optional[List[int]],
        stride: List[int], padding: List[int], dilation: List[int],
        groups: int): ...


def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = ...): ...


def zero_dim_tensor(input: Any): ...


def multiply_integers(li: List[int]): ...


def arange_end(end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any): ...


def arange_start(
    start: number, end: number, inp0: Any, inp1: Any, inp2: Any,
        inp3: Any): ...


def arange_start_step(
    start: number, end: number, step: number, inp0: Any, inp1: Any,
        inp2: Any, inp3: Any): ...


def permute(input: List[int], dims: List[int]): ...


def flatten(input: List[int], start_dim: int, end_dim: int): ...


def nonzero_lower_bound(input: List[int]): ...


def nonzero_upper_bound(input: List[int]): ...


def argmax(
    self, dim: Optional[int] = ..., keepdim: bool = ...) -> List[int]: ...


def bmm(self, mat2: List[int]) -> List[int]: ...


def topk(self, k: int, dim: int = ...) -> Tuple[List[int], List[int]]: ...


def nll_loss_forward(
    self, target: List[int], weight: Optional[List[int]],
        reduction: int) -> Tuple[List[int], List[int]]: ...


def native_layer_norm(
    input: List[int], normalized_shape: List[int]) -> Tuple[List[int],
        List[int], List[int]]: ...


def native_batch_norm(
    input: List[int], weight: Optional[List[int]], bias: Optional[List[int]],
        running_mean: Optional[List[int]], running_var: Optional[List[int]],
        training: bool) -> Tuple[List[int], List[int], List[int]]: ...


ScriptFn: Incomplete
shape_compute_graph_mapping: Dict[str, ScriptFn]
bounded_compute_graph_mapping: Dict[str, Tuple[ScriptFn, ScriptFn]]
script_func_map: Dict[Callable, ScriptFn]


def process_func(func: Callable): ...


def add_shape_compute_mapping(operator_schema: str, func: Callable): ...


def add_bounded_compute_mapping(
    operator_schema: str, lower_bound_func: Callable,
        upper_bound_func: Callable): ...
