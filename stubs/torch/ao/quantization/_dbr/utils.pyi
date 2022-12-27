# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from _typeshed import Incomplete
from torch.quantization import FakeQuantizeBase as FakeQuantizeBase
from torch.quantization import (
    is_activation_post_process as is_activation_post_process,
)
from torch.quantization import ObserverBase as ObserverBase

from ..qconfig import QConfigAny as QConfigAny
from ..qconfig_dict_utils import (
    maybe_adjust_qconfig_for_module_type_or_name as maybe_adjust_qconfig_for_module_type_or_name,
)
from .mappings import add_and_mul_ops as add_and_mul_ops
from .mappings import conv_ops as conv_ops
from .mappings import fp32_to_int8_fun_mapping as fp32_to_int8_fun_mapping
from .mappings import (
    functions_supported_by_quantization as functions_supported_by_quantization,
)
from .mappings import (
    functions_supported_by_quantization_preserves_dtype as functions_supported_by_quantization_preserves_dtype,
)
from .mappings import (
    module_types_supported_by_quantization as module_types_supported_by_quantization,
)
from .mappings import (
    module_types_supported_by_quantization_preserves_dtype as module_types_supported_by_quantization_preserves_dtype,
)


toq: Incomplete


class QTensorInfo:
    id: int
    orig_dtype: torch.dtype
    inf_dtype: torch.dtype
    def __init__(self, id, orig_dtype, inf_dtype) -> None: ...


class FusionInfo:
    pattern: Tuple[Callable, ...]
    replacement_type_this_element: Callable
    is_first_element: bool
    is_last_element: bool

    def __init__(
        self, pattern, replacement_type_this_element, is_first_element,
        is_last_element) -> None: ...


class SeenQOpInfo:
    idx: int
    type: Callable
    type_is_module: bool
    fqn: str
    input_tensor_infos: List[Optional[QTensorInfo]]
    output_tensor_infos: List[QTensorInfo]
    packable_tensor_idx_to_name: Dict[int, Optional[str]]
    packable_nontensor_idx_to_arg: Dict[int, Any]
    packable_tensor_kwarg_name_to_name: Dict[str, Optional[str]]
    op_packing_only_uses_module_attributes: bool
    qconfig: QConfigAny
    fusion_info: Optional[FusionInfo]
    is_reference_op_at_inference: bool

    def __init__(
        self, idx, type, type_is_module, fqn, input_tensor_infos,
        output_tensor_infos, packable_tensor_idx_to_name,
        packable_nontensor_idx_to_arg, packable_tensor_kwarg_name_to_name,
        op_packing_only_uses_module_attributes, qconfig, fusion_info,
        is_reference_op_at_inference) -> None: ...


class SeenNonQOpInfo:
    type: Callable
    input_tensor_infos: List[Optional[QTensorInfo]]
    output_tensor_infos: List[QTensorInfo]

    def __init__(
        self, type, input_tensor_infos, output_tensor_infos) -> None: ...


class OpQuantizeabilityType(enum.Enum):
    QUANTIZEABLE: int
    NOT_QUANTIZEABLE: int


def op_needs_quantization(op: Callable) -> bool: ...


class ObserverWrapper(torch.nn.Identity):
    child: Incomplete
    dtype: Incomplete
    def __init__(self, child) -> None: ...


def wrap_observers_in_placeholders(module: torch.nn.Module) -> None: ...


def unwrap_observers_from_placeholders(module: torch.nn.Module) -> None: ...


def trace_with_inputs(
    model: torch.nn.Module, example_args: Tuple[Any]) -> None: ...


def is_leaf(
    m: torch.nn.Module, prepare_custom_config_dict: Optional[Dict[str,
                    Any]]) -> bool: ...


class FuncOutputObsType(enum.Enum):
    NONE: int
    NEW_OBS: int
    REUSES_FIRST_INPUT_OBS: int


def get_func_output_obs_type(
    seen_q_op_info: SeenQOpInfo) -> FuncOutputObsType: ...


def converted_func_needs_scale_zp(seen_q_op_info: SeenQOpInfo) -> bool: ...


class FuncOutputDTypeType(enum.Enum):
    DTYPE_DEPENDS_ON_QCONFIG: int
    DTYPE_EQUALS_INPUT_DTYPE: int
    DTYPE_DEFAULT_BC_UNSUPPORTED_SYNTAX: int


def get_func_output_dtype_type(
    seen_q_op_info: SeenQOpInfo) -> FuncOutputDTypeType: ...


def get_weight_argument_info(op: Callable) -> Optional[Tuple[int, str]]: ...


def get_op_packing_only_uses_module_attributes(
    op: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any],
    module: torch.nn.Module) -> bool: ...


def get_quantized_op(
    seen_q_op_info: SeenQOpInfo, idx_to_seen_q_op_infos: Dict[int,
            SeenQOpInfo]) -> Optional[Callable]: ...


def get_input_observed_arg_idxs(
    op_type: Callable, op_type_is_module: bool) -> Optional[List[int]]: ...


def get_packable_tensor_arg_idxs(op: Callable) -> Optional[List[int]]: ...


def get_packable_tensor_kwarg_names(op: Callable) -> Optional[List[str]]: ...


def get_param_name(module: torch.nn.Module, arg: Any) -> Optional[str]: ...


def get_packable_nontensor_arg_idxs(op: Callable) -> Optional[List[int]]: ...


def get_packable_arg_idxs(op: Callable) -> Optional[List[int]]: ...


def get_weight_arg_idx(op: Callable) -> Optional[int]: ...


def iterate_and_apply(
    args: Any, flattened_tensor_infos: List[Optional[QTensorInfo]],
    func: Callable,
    flattened_tensor_infos_idx: Incomplete | None = ...) -> Any: ...


def get_producer_of_seen_q_op_info(
    idx_to_seen_q_op_info: Dict[int, SeenQOpInfo],
    cur_seen_q_op_info: SeenQOpInfo) -> Optional[SeenQOpInfo]: ...


def get_users_of_seen_q_op_info(
    idx_to_seen_q_op_info: Dict[int, SeenQOpInfo],
    cur_seen_q_op_info: SeenQOpInfo) -> List[SeenQOpInfo]: ...


class HookType(enum.Enum):
    OP_HOOKS: int
    MODULE_IO_HOOKS: int
    ARG_DEQUANTS: int
    NONE: int


def get_torch_function_hook_type(
    parent_module: Optional[torch.nn.Module], func: Callable) -> HookType: ...


def get_module_hook_type(
    parent_module: Optional[
            torch.nn.Module], cur_module: torch.nn.Module) -> HookType: ...


def clone_detach_tensor_without_dispatch(x: torch.Tensor) -> torch.Tensor: ...


def get_input_args_quant_dequant_info(
    seen_q_op_info: SeenQOpInfo, tensor_id_to_scale_zp: Dict[int,
            Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[Optional[Tuple[
                                float, int, torch.dtype]]], List[
                bool], bool]: ...


def get_cur_qconfig(
    qconfig_dict: Dict[
            str, Any], cur_fqn: str, cur_op_type: Callable) -> Optional[
        QConfigAny]: ...


class AutoQuantizationStateModuleDict(torch.nn.ModuleDict):
    ...


def get_fqn_valid_for_module_dict_key(fqn: str) -> str: ...
