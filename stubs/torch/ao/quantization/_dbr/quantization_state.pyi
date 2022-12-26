from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from _typeshed import Incomplete
from torch.ao.quantization.utils import (
    activation_is_int32_quantized as activation_is_int32_quantized,
)

from .function_fusion import (
    get_seen_q_op_info_of_end_of_fusion as get_seen_q_op_info_of_end_of_fusion,
)
from .function_fusion import (
    get_seen_q_op_info_of_start_of_fusion as get_seen_q_op_info_of_start_of_fusion,
)
from .function_fusion import match_fusion_patterns as match_fusion_patterns
from .mappings import conv_ops as conv_ops
from .mappings import ops_are_related as ops_are_related
from .utils import (
    clone_detach_tensor_without_dispatch as clone_detach_tensor_without_dispatch,
)
from .utils import (
    converted_func_needs_scale_zp as converted_func_needs_scale_zp,
)
from .utils import FuncOutputDTypeType as FuncOutputDTypeType
from .utils import FuncOutputObsType as FuncOutputObsType
from .utils import get_cur_qconfig as get_cur_qconfig
from .utils import get_func_output_dtype_type as get_func_output_dtype_type
from .utils import get_func_output_obs_type as get_func_output_obs_type
from .utils import (
    get_input_args_quant_dequant_info as get_input_args_quant_dequant_info,
)
from .utils import get_input_observed_arg_idxs as get_input_observed_arg_idxs
from .utils import (
    get_op_packing_only_uses_module_attributes as get_op_packing_only_uses_module_attributes,
)
from .utils import get_packable_arg_idxs as get_packable_arg_idxs
from .utils import (
    get_packable_nontensor_arg_idxs as get_packable_nontensor_arg_idxs,
)
from .utils import get_packable_tensor_arg_idxs as get_packable_tensor_arg_idxs
from .utils import (
    get_packable_tensor_kwarg_names as get_packable_tensor_kwarg_names,
)
from .utils import get_param_name as get_param_name
from .utils import get_quantized_op as get_quantized_op
from .utils import get_weight_arg_idx as get_weight_arg_idx
from .utils import iterate_and_apply as iterate_and_apply
from .utils import op_needs_quantization as op_needs_quantization
from .utils import OpQuantizeabilityType as OpQuantizeabilityType
from .utils import QTensorInfo as QTensorInfo
from .utils import SeenNonQOpInfo as SeenNonQOpInfo
from .utils import SeenQOpInfo as SeenQOpInfo


OpConvertInfo: Incomplete

class AutoQuantizationState(torch.nn.Module):
    idx: int
    qconfig_dict: Incomplete
    fqn: Incomplete
    tensor_id_to_observer: Incomplete
    idx_to_seen_q_op_infos: Incomplete
    seen_nonq_op_infos: Incomplete
    output_qtensor_infos: Incomplete
    input_dtypes: Incomplete
    output_dtypes: Incomplete
    idx_to_packed_weight_name: Incomplete
    tensor_id_to_scale_zp: Incomplete
    log_op_outputs: bool
    op_outputs: Incomplete
    logging_model_name: Incomplete
    idx_to_op_convert_info: Incomplete
    needs_dtype_transform_on_outputs: bool
    def __init__(self, qconfig_dict: Dict[str, Any], fqn: str, input_dtypes: Any = ..., output_dtypes: Any = ...) -> None: ...
    def get_extra_state(self): ...
    def set_extra_state(self, state) -> None: ...
    def has_at_least_one_seen_q_op_info(self) -> bool: ...
    def validate_is_at_last_seen_idx(self) -> None: ...
    def extra_repr(self) -> str: ...
    def get_cur_output_inf_dtype(self): ...
    def reset_to_new_call(self) -> None: ...
    def cur_op_needs_hooks(self, cur_op: Callable) -> bool: ...
    def validate_cur_op(self, cur_op: Callable) -> None: ...
    def mark_cur_op_complete(self, cur_op: Callable) -> None: ...
    def first_call_outputs_prepare_hook(self, outputs: Any, qtensor_id: List[int]) -> Any: ...
    def outputs_prepare_hook(self, outputs: Any) -> Any: ...
    def outputs_convert_hook(self, outputs: Any) -> Any: ...
    def get_output_qtensor_infos(self) -> List[Optional[QTensorInfo]]: ...
    def get_output_dtypes(self) -> Any: ...
    def first_call_op_prepare_before_hook(self, op: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], qtensor_id: List[int], fqn: str, root_module: torch.nn.Module, op_quantizeability_type: OpQuantizeabilityType) -> Tuple[Tuple[Any, ...], Dict[str, Any]]: ...
    def op_prepare_before_hook(self, op: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]: ...
    def first_call_op_prepare_after_hook(self, op: Callable, output: Any, args: Tuple[Any, ...], qtensor_id: List[int], op_quantizeability_type: OpQuantizeabilityType) -> Any: ...
    def op_prepare_after_hook(self, op: Callable, output: Any, args: Tuple[Any, ...], global_op_idx: List[int]) -> Any: ...
    def op_convert_before_hook(self, op: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], root_module: torch.nn.Module) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]: ...
    def op_convert_after_hook(self, op: Callable, output, global_op_idx: List[int]) -> Any: ...
    def get_op_convert_info(self, op: Callable) -> OpConvertInfo: ...
    def calculate_op_convert_info(self, seen_q_op_info: SeenQOpInfo) -> OpConvertInfo: ...
    def set_needs_dtype_transform_on_outputs(self) -> None: ...
    def match_fusion_patterns(self) -> None: ...
    def insert_observers(self, root_module: torch.nn.Module): ...
    def get_output_observer_from_fqn(self, fqn: str) -> Optional[torch.nn.Module]: ...
    def forward(self, x) -> None: ...