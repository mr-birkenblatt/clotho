# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import builtins
from collections.abc import Generator as Generator
from math import e as e
from math import inf as inf
from math import nan as nan
from math import pi as pi
from typing import Type as Type
from typing import Union

from torch._C import *
from torch._C._VariableFunctions import *
from torch.autograd import enable_grad as enable_grad
from torch.autograd import inference_mode as inference_mode
from torch.autograd import no_grad as no_grad

from ._lobpcg import lobpcg as lobpcg
from ._tensor import Tensor as Tensor
from ._tensor_str import set_printoptions as set_printoptions
from .functional import *
from .random import get_rng_state as get_rng_state
from .random import initial_seed as initial_seed
from .random import manual_seed as manual_seed
from .random import seed as seed
from .random import set_rng_state as set_rng_state
from .serialization import load as load
from .serialization import save as save
from .storage import _LegacyStorage
from .storage import _TypedStorage as _TypedStorage


def typename(o): ...


def is_tensor(obj): ...


def is_storage(obj): ...


def set_default_tensor_type(t) -> None: ...


def use_deterministic_algorithms(mode, *, warn_only: bool = ...) -> None: ...


def are_deterministic_algorithms_enabled(): ...


def is_deterministic_algorithms_warn_only_enabled(): ...


def set_deterministic_debug_mode(
    debug_mode: Union[builtins.int, str]) -> None: ...


def get_deterministic_debug_mode() -> builtins.int: ...


def get_float32_matmul_precision() -> builtins.str: ...


def set_float32_matmul_precision(precision) -> None: ...


def set_warn_always(b) -> None: ...


def is_warn_always_enabled(): ...


class ByteStorage(_LegacyStorage):
    def dtype(self): ...


class DoubleStorage(_LegacyStorage):
    def dtype(self): ...


class FloatStorage(_LegacyStorage):
    def dtype(self): ...


class HalfStorage(_LegacyStorage):
    def dtype(self): ...


class LongStorage(_LegacyStorage):
    def dtype(self): ...


class IntStorage(_LegacyStorage):
    def dtype(self): ...


class ShortStorage(_LegacyStorage):
    def dtype(self): ...


class CharStorage(_LegacyStorage):
    def dtype(self): ...


class BoolStorage(_LegacyStorage):
    def dtype(self): ...


class BFloat16Storage(_LegacyStorage):
    def dtype(self): ...


class ComplexDoubleStorage(_LegacyStorage):
    def dtype(self): ...


class ComplexFloatStorage(_LegacyStorage):
    def dtype(self): ...


class QUInt8Storage(_LegacyStorage):
    def dtype(self): ...


class QInt8Storage(_LegacyStorage):
    def dtype(self): ...


class QInt32Storage(_LegacyStorage):
    def dtype(self): ...


class QUInt4x2Storage(_LegacyStorage):
    def dtype(self): ...


class QUInt2x4Storage(_LegacyStorage):
    def dtype(self): ...


legacy_contiguous_format = contiguous_format

# Names in __all__ with no definition:
#   AVG
#   AggregationType
#   AliasDb
#   AnyType
#   Argument
#   ArgumentSpec
#   BenchmarkConfig
#   BenchmarkExecutionStats
#   Block
#   BoolTensor
#   BoolType
#   BufferDict
#   ByteTensor
#   CONV_BN_FUSION
#   CallStack
#   Capsule
#   CharTensor
#   ClassType
#   Code
#   CompilationUnit
#   CompleteArgumentSpec
#   ComplexType
#   ConcreteModuleType
#   ConcreteModuleTypeBuilder
#   DeepCopyMemoTable
#   DeserializationStorageContext
#   DeviceObjType
#   DictType
#   DisableTorchFunction
#   DoubleTensor
#   EnumType
#   ErrorReport
#   ExecutionPlan
#   FUSE_ADD_RELU
#   FatalError
#   FileCheck
#   FloatTensor
#   FloatType
#   FunctionSchema
#   Future
#   FutureType
#   Generator
#   Gradient
#   Graph
#   GraphExecutorState
#   HOIST_CONV_PACKED_PARAMS
#   INSERT_FOLD_PREPACK_OPS
#   IODescriptor
#   InferredType
#   IntTensor
#   IntType
#   InterfaceType
#   JITException
#   ListType
#   LiteScriptModule
#   LockingLogger
#   LongTensor
#   MobileOptimizerType
#   ModuleDict
#   Node
#   NoneType
#   NoopLogger
#   NumberType
#   OperatorInfo
#   OptionalType
#   ParameterDict
#   PyObjectType
#   PyTorchFileReader
#   PyTorchFileWriter
#   REMOVE_DROPOUT
#   RRefType
#   SUM
#   ScriptClass
#   ScriptClassFunction
#   ScriptDict
#   ScriptDictIterator
#   ScriptDictKeyIterator
#   ScriptFunction
#   ScriptList
#   ScriptListIterator
#   ScriptMethod
#   ScriptModule
#   ScriptModuleSerializer
#   ScriptObject
#   ScriptObjectProperty
#   SerializationStorageContext
#   ShortTensor
#   Size
#   StaticModule
#   Stream
#   StreamObjType
#   StringType
#   SymIntType
#   TensorType
#   ThroughputBenchmark
#   TracingState
#   TupleType
#   UnionType
#   Use
#   Value
#   abs
#   abs_
#   absolute
#   acos
#   acos_
#   acosh
#   acosh_
#   adaptive_avg_pool1d
#   adaptive_max_pool1d
#   add
#   addbmm
#   addcdiv
#   addcmul
#   addmm
#   addmv
#   addmv_
#   addr
#   adjoint
#   affine_grid_generator
#   alias_copy
#   align_tensors
#   all
#   allclose
#   alpha_dropout
#   alpha_dropout_
#   amax
#   amin
#   aminmax
#   angle
#   any
#   arange
#   arccos
#   arccos_
#   arccosh
#   arccosh_
#   arcsin
#   arcsin_
#   arcsinh
#   arcsinh_
#   arctan
#   arctan2
#   arctan_
#   arctanh
#   arctanh_
#   argmax
#   argmin
#   argsort
#   argwhere
#   as_strided
#   as_strided_
#   as_strided_copy
#   as_tensor
#   asarray
#   asin
#   asin_
#   asinh
#   asinh_
#   atan
#   atan2
#   atan_
#   atanh
#   atanh_
#   atleast_1d
#   atleast_2d
#   atleast_3d
#   autocast_decrement_nesting
#   autocast_increment_nesting
#   avg_pool1d
#   baddbmm
#   bartlett_window
#   batch_norm
#   batch_norm_backward_elemt
#   batch_norm_backward_reduce
#   batch_norm_elemt
#   batch_norm_gather_stats
#   batch_norm_gather_stats_with_counts
#   batch_norm_stats
#   batch_norm_update_stats
#   bernoulli
#   bilinear
#   binary_cross_entropy_with_logits
#   bincount
#   binomial
#   bitwise_and
#   bitwise_left_shift
#   bitwise_not
#   bitwise_or
#   bitwise_right_shift
#   bitwise_xor
#   blackman_window
#   block_diag
#   bmm
#   broadcast_tensors
#   broadcast_to
#   bucketize
#   can_cast
#   cartesian_prod
#   cat
#   ccol_indices_copy
#   cdist
#   ceil
#   ceil_
#   celu
#   celu_
#   chain_matmul
#   channel_shuffle
#   cholesky
#   cholesky_inverse
#   cholesky_solve
#   choose_qparams_optimized
#   chunk
#   chunk
#   clamp
#   clamp_
#   clamp_max
#   clamp_max_
#   clamp_min
#   clamp_min_
#   clear_autocast_cache
#   clip
#   clip_
#   clone
#   col_indices_copy
#   column_stack
#   combinations
#   complex
#   concat
#   conj
#   conj_physical
#   conj_physical_
#   constant_pad_nd
#   conv1d
#   conv2d
#   conv3d
#   conv_tbc
#   conv_transpose1d
#   conv_transpose2d
#   conv_transpose3d
#   convolution
#   copysign
#   corrcoef
#   cos
#   cos_
#   cosh
#   cosh_
#   cosine_embedding_loss
#   cosine_similarity
#   count_nonzero
#   cov
#   cpp
#   cross
#   crow_indices_copy
#   ctc_loss
#   cudnn_affine_grid_generator
#   cudnn_batch_norm
#   cudnn_convolution
#   cudnn_convolution_add_relu
#   cudnn_convolution_relu
#   cudnn_convolution_transpose
#   cudnn_grid_sampler
#   cudnn_is_acceptable
#   cummax
#   cummin
#   cumprod
#   cumsum
#   cumulative_trapezoid
#   default_generator
#   deg2rad
#   deg2rad_
#   dequantize
#   det
#   detach
#   detach_
#   detach_copy
#   device
#   diag
#   diag_embed
#   diagflat
#   diagonal
#   diagonal_copy
#   diagonal_scatter
#   diff
#   digamma
#   dist
#   div
#   divide
#   dot
#   dropout
#   dropout_
#   dsmm
#   dsplit
#   dstack
#   dtype
#   eig
#   einsum
#   embedding
#   embedding_bag
#   embedding_renorm_
#   empty
#   empty_like
#   empty_quantized
#   empty_strided
#   eq
#   equal
#   erf
#   erf_
#   erfc
#   erfc_
#   erfinv
#   exp
#   exp2
#   exp2_
#   exp_
#   expand_copy
#   expm1
#   expm1_
#   eye
#   fake_quantize_per_channel_affine
#   fake_quantize_per_tensor_affine
#   fbgemm_linear_fp16_weight
#   fbgemm_linear_fp16_weight_fp32_activation
#   fbgemm_linear_int8_weight
#   fbgemm_linear_int8_weight_fp32_activation
#   fbgemm_linear_quantize_weight
#   fbgemm_pack_gemm_matrix_fp16
#   fbgemm_pack_quantized_matrix
#   feature_alpha_dropout
#   feature_alpha_dropout_
#   feature_dropout
#   feature_dropout_
#   fill
#   fill_
#   finfo
#   fix
#   fix_
#   flatten
#   flip
#   fliplr
#   flipud
#   float_power
#   floor
#   floor_
#   floor_divide
#   fmax
#   fmin
#   fmod
#   fork
#   frac
#   frac_
#   frexp
#   frobenius_norm
#   from_file
#   from_numpy
#   frombuffer
#   full
#   full_like
#   fused_moving_avg_obs_fake_quant
#   gather
#   gcd
#   gcd_
#   ge
#   geqrf
#   ger
#   get_autocast_cpu_dtype
#   get_autocast_gpu_dtype
#   get_default_dtype
#   get_device
#   get_num_interop_threads
#   get_num_threads
#   gradient
#   greater
#   greater_equal
#   grid_sampler
#   grid_sampler_2d
#   grid_sampler_3d
#   group_norm
#   gru
#   gru_cell
#   gt
#   hamming_window
#   hann_window
#   hardshrink
#   has_cuda
#   has_cudnn
#   has_lapack
#   has_mkl
#   has_mkldnn
#   has_mps
#   has_openmp
#   has_spectral
#   heaviside
#   hinge_embedding_loss
#   histc
#   histogram
#   histogramdd
#   hsmm
#   hsplit
#   hspmm
#   hstack
#   hypot
#   i0
#   i0_
#   igamma
#   igammac
#   iinfo
#   imag
#   import_ir_module
#   import_ir_module_from_buffer
#   index_add
#   index_copy
#   index_fill
#   index_put
#   index_put_
#   index_reduce
#   index_select
#   indices_copy
#   init_num_threads
#   inner
#   instance_norm
#   int_repr
#   inverse
#   is_anomaly_enabled
#   is_autocast_cache_enabled
#   is_autocast_cpu_enabled
#   is_autocast_enabled
#   is_complex
#   is_conj
#   is_distributed
#   is_floating_point
#   is_grad_enabled
#   is_inference
#   is_inference_mode_enabled
#   is_neg
#   is_nonzero
#   is_same_size
#   is_signed
#   is_vulkan_available
#   isclose
#   isfinite
#   isin
#   isinf
#   isnan
#   isneginf
#   isposinf
#   isreal
#   istft
#   kaiser_window
#   kl_div
#   kron
#   kthvalue
#   layer_norm
#   layout
#   lcm
#   lcm_
#   ldexp
#   ldexp_
#   le
#   lerp
#   less
#   less_equal
#   lgamma
#   linspace
#   log
#   log10
#   log10_
#   log1p
#   log1p_
#   log2
#   log2_
#   log_
#   log_softmax
#   logaddexp
#   logaddexp2
#   logcumsumexp
#   logdet
#   logical_and
#   logical_not
#   logical_or
#   logical_xor
#   logit
#   logit_
#   logspace
#   logsumexp
#   lstm
#   lstm_cell
#   lstsq
#   lt
#   lu_solve
#   lu_unpack
#   margin_ranking_loss
#   masked_fill
#   masked_scatter
#   masked_select
#   matmul
#   matmul
#   matrix_exp
#   matrix_power
#   matrix_rank
#   max
#   max_pool1d
#   max_pool1d_with_indices
#   max_pool2d
#   max_pool3d
#   maximum
#   mean
#   median
#   memory_format
#   merge_type_from_type_comment
#   meshgrid
#   min
#   minimum
#   miopen_batch_norm
#   miopen_convolution
#   miopen_convolution_transpose
#   miopen_depthwise_convolution
#   miopen_rnn
#   mkldnn_adaptive_avg_pool2d
#   mkldnn_convolution
#   mkldnn_linear_backward_weights
#   mkldnn_max_pool2d
#   mkldnn_max_pool3d
#   mm
#   mode
#   moveaxis
#   movedim
#   msort
#   mul
#   multinomial
#   multiply
#   mv
#   mvlgamma
#   nan_to_num
#   nan_to_num_
#   nanmean
#   nanmedian
#   nanquantile
#   nansum
#   narrow
#   narrow_copy
#   native_batch_norm
#   native_channel_shuffle
#   native_dropout
#   native_group_norm
#   native_layer_norm
#   native_norm
#   ne
#   neg
#   neg_
#   negative
#   negative_
#   nested_tensor
#   nextafter
#   nonzero
#   norm
#   norm_except_dim
#   normal
#   not_equal
#   nuclear_norm
#   numel
#   ones
#   ones_like
#   orgqr
#   ormqr
#   outer
#   pairwise_distance
#   parse_ir
#   parse_schema
#   parse_type_comment
#   pdist
#   permute
#   permute_copy
#   pinverse
#   pixel_shuffle
#   pixel_unshuffle
#   poisson
#   poisson_nll_loss
#   polar
#   polygamma
#   positive
#   pow
#   prelu
#   prod
#   promote_types
#   put
#   q_per_channel_axis
#   q_per_channel_scales
#   q_per_channel_zero_points
#   q_scale
#   q_zero_point
#   qr
#   qscheme
#   quantile
#   quantize_per_channel
#   quantize_per_tensor
#   quantize_per_tensor_dynamic
#   quantized_batch_norm
#   quantized_gru_cell
#   quantized_lstm_cell
#   quantized_max_pool1d
#   quantized_max_pool2d
#   quantized_rnn_relu_cell
#   quantized_rnn_tanh_cell
#   rad2deg
#   rad2deg_
#   rand
#   rand
#   rand_like
#   randint
#   randint_like
#   randn
#   randn
#   randn_like
#   randperm
#   range
#   ravel
#   read_vitals
#   real
#   reciprocal
#   reciprocal_
#   relu
#   relu_
#   remainder
#   renorm
#   repeat_interleave
#   reshape
#   resize_as_
#   resize_as_sparse_
#   resolve_conj
#   resolve_neg
#   result_type
#   rnn_relu
#   rnn_relu_cell
#   rnn_tanh
#   rnn_tanh_cell
#   roll
#   rot90
#   round
#   round_
#   row_indices_copy
#   row_stack
#   rrelu
#   rrelu_
#   rsqrt
#   rsqrt_
#   rsub
#   saddmm
#   scalar_tensor
#   scatter
#   scatter_add
#   scatter_reduce
#   searchsorted
#   segment_reduce
#   select
#   select_copy
#   select_scatter
#   selu
#   selu_
#   set_anomaly_enabled
#   set_autocast_cache_enabled
#   set_autocast_cpu_dtype
#   set_autocast_cpu_enabled
#   set_autocast_enabled
#   set_autocast_gpu_dtype
#   set_flush_denormal
#   set_num_interop_threads
#   set_num_threads
#   set_vital
#   sgn
#   sigmoid
#   sigmoid_
#   sign
#   signbit
#   sin
#   sin_
#   sinc
#   sinc_
#   sinh
#   sinh_
#   slice_copy
#   slice_scatter
#   slogdet
#   smm
#   softmax
#   sort
#   sparse_bsc_tensor
#   sparse_bsr_tensor
#   sparse_compressed_tensor
#   sparse_coo_tensor
#   sparse_csc_tensor
#   sparse_csr_tensor
#   split
#   split
#   split_copy
#   split_with_sizes
#   split_with_sizes_copy
#   spmm
#   sqrt
#   sqrt_
#   square
#   square_
#   squeeze
#   squeeze_copy
#   sspaddmm
#   stack
#   stack
#   std
#   std_mean
#   stft
#   sub
#   subtract
#   sum
#   svd
#   swapaxes
#   swapdims
#   symeig
#   t
#   t_copy
#   take
#   take_along_dim
#   tan
#   tan_
#   tanh
#   tanh_
#   tensor
#   tensor_split
#   tensordot
#   threshold
#   threshold_
#   tile
#   topk
#   trace
#   transpose
#   transpose_copy
#   trapezoid
#   trapz
#   triangular_solve
#   tril
#   tril_indices
#   triplet_margin_loss
#   triu
#   triu_indices
#   true_divide
#   trunc
#   trunc_
#   unbind
#   unbind_copy
#   unfold_copy
#   unify_type_list
#   unique_consecutive
#   unsafe_chunk
#   unsafe_split
#   unsafe_split_with_sizes
#   unsqueeze
#   unsqueeze_copy
#   values_copy
#   vander
#   var
#   var_mean
#   vdot
#   view_as_complex
#   view_as_complex_copy
#   view_as_real
#   view_as_real_copy
#   view_copy
#   vitals_enabled
#   vsplit
#   vstack
#   wait
#   where
#   xlogy
#   xlogy_
#   zero_
#   zeros
#   zeros_like
