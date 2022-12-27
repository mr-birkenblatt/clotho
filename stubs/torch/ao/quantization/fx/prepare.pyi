# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from _typeshed import Incomplete
from torch.ao.quantization.quantization_types import NodePattern as NodePattern
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.ao.quantization.quantize import convert as convert
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node
from torch.fx.node import Argument as Argument

from ..backend_config import (
    get_native_backend_config_dict as get_native_backend_config_dict,
)
from ..backend_config.utils import (
    get_fusion_pattern_to_root_node_getter as get_fusion_pattern_to_root_node_getter,
)
from ..backend_config.utils import (
    get_module_to_qat_module as get_module_to_qat_module,
)
from ..backend_config.utils import (
    get_pattern_to_dtype_configs as get_pattern_to_dtype_configs,
)
from ..backend_config.utils import (
    get_pattern_to_input_type_to_index as get_pattern_to_input_type_to_index,
)
from ..observer import ObserverBase as ObserverBase
from ..qconfig import is_reuse_input_qconfig as is_reuse_input_qconfig
from ..qconfig import QConfigAny as QConfigAny
from ..qconfig_dict_utils import (
    convert_dict_to_ordered_dict as convert_dict_to_ordered_dict,
)
from ..qconfig_dict_utils import (
    get_flattened_qconfig_dict as get_flattened_qconfig_dict,
)
from ..qconfig_dict_utils import (
    update_qconfig_for_qat as update_qconfig_for_qat,
)
from ..quantize import propagate_qconfig_ as propagate_qconfig_
from ..utils import (
    activation_is_int8_quantized as activation_is_int8_quantized,
)
from ..utils import (
    activation_is_statically_quantized as activation_is_statically_quantized,
)
from ..utils import get_qconfig_dtypes as get_qconfig_dtypes
from ..utils import (
    get_swapped_custom_module_class as get_swapped_custom_module_class,
)
from ._equalize import is_equalization_observer as is_equalization_observer
from ._equalize import node_supports_equalization as node_supports_equalization
from .backend_config_utils import (
    get_pattern_to_quantize_handlers as get_pattern_to_quantize_handlers,
)
from .graph_module import ObservedGraphModule as ObservedGraphModule
from .graph_module import (
    ObservedStandaloneGraphModule as ObservedStandaloneGraphModule,
)
from .match_utils import find_matches as find_matches
from .pattern_utils import MatchResult as MatchResult
from .pattern_utils import sorted_patterns_dict as sorted_patterns_dict
from .qconfig_utils import generate_qconfig_map as generate_qconfig_map
from .qconfig_utils import (
    get_standalone_module_configs as get_standalone_module_configs,
)
from .qconfig_utils import (
    update_qconfig_for_fusion as update_qconfig_for_fusion,
)
from .quantization_patterns import QuantizeHandler as QuantizeHandler
from .utils import (
    all_node_args_have_no_tensors as all_node_args_have_no_tensors,
)
from .utils import assert_and_get_unique_device as assert_and_get_unique_device
from .utils import BIAS_INDEX_DICT as BIAS_INDEX_DICT
from .utils import get_custom_module_class_keys as get_custom_module_class_keys
from .utils import (
    get_new_attr_name_with_prefix as get_new_attr_name_with_prefix,
)
from .utils import (
    get_non_observable_arg_indexes_and_types as get_non_observable_arg_indexes_and_types,
)
from .utils import NON_QUANTIZABLE_WEIGHT_OPS as NON_QUANTIZABLE_WEIGHT_OPS
from .utils import WEIGHT_INDEX_DICT as WEIGHT_INDEX_DICT


DO_NOT_OBS_DTYPE_LIST: Incomplete


def is_activation_post_process_node(
    node: Node, modules: Dict[str, torch.nn.Module]) -> bool: ...


def node_arg_is_weight(node: Node, arg: Any) -> bool: ...


def node_arg_is_bias(node: Node, arg: Any) -> bool: ...


def is_input_arg_dtype_supported_by_backend(
    arg: Argument, node: Node, node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    dtype_config: Dict[str, torch.dtype]) -> bool: ...


def is_output_dtype_supported_by_backend(
    node: Node, node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    dtype_config: Dict[str, torch.dtype]) -> bool: ...


def is_observer_in_same_graph(node, modules, node_name_to_target_dtype): ...


def is_pattern_dtype_config_supported_by_backend(
    pattern: Optional[Pattern], matched_node_pattern: Optional[NodePattern],
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    backend_config_dict: Optional[Dict[str, Any]]) -> bool: ...


def prepare_get_standalone_module_configs(
    node: Node, modules: Dict[str, torch.nn.Module],
    prepare_custom_config_dict: Dict[str, Any], parent_qconfig: QConfigAny,
    parent_backend_config_dict: Optional[Dict[str, Any]]) -> Tuple[Dict[
                str, Any], Dict[str, Any], Dict[str, Any]]: ...


def qat_swap_modules(
    root: torch.nn.Module, module_to_qat_module: Dict[
            Callable, Callable]) -> None: ...


def add_matched_node_name_to_set(
    matched_node_pattern: NodePattern, s: Set[str]): ...


def insert_observer(
    node: Node, observer: ObserverBase, model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module], graph: Graph) -> Node: ...


def get_target_activation_dtype_for_node(
    node: Node, qconfig: QConfigAny, inputs_seen_counter: int,
    outputs_seen_counter: int, input_quantized_idxs: List[int],
    output_quantized_idxs: List[int], qhandler: Optional[QuantizeHandler],
    modules: Dict[str, torch.nn.Module],
    cache_for_no_tensor_check: Dict[Node, bool]) -> Dict[str, Optional[Union[
                        torch.dtype, type]]]: ...


def get_arg_target_dtype_as_output(
    arg: Node, modules: Dict[str, torch.nn.Module],
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]]) -> Optional[Union[
                torch.dtype, type]]: ...


def get_arg_target_dtype_as_input_to_node(
    arg: Node, node: Node, modules: Dict[str, torch.nn.Module],
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]]) -> Optional[Union[
                torch.dtype, type]]: ...


def get_arg_target_compute_dtype_as_input_to_node(
    arg: Node, node: Node, modules: Dict[str, torch.nn.Module],
    node_name_to_target_dtype: Dict[str, Dict[str, Union[torch.dtype, type,
                            None]]]) -> Union[torch.dtype, type, None]: ...


def maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any], arg: Argument, qconfig: QConfigAny,
    model: torch.nn.Module, modules: Dict[str, torch.nn.Module],
    graph: Graph, node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
    backend_config_dict: Optional[Dict[str, Any]]) -> Argument: ...


def maybe_insert_input_observers_for_node(
    node: Node, qconfig: QConfigAny, model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module], graph: Graph,
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
    backend_config_dict: Optional[Dict[str, Any]]) -> None: ...


def maybe_insert_input_equalization_observers_for_node(
    node: Node, equalization_qconfig: Any, model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module], graph: Graph,
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[
                                    torch.dtype, type]]]],
    is_branch: bool) -> None: ...


def maybe_insert_output_observer_for_node(
    node: Node, model: torch.nn.Module, modules: Dict[str, torch.nn.Module],
    graph: Graph, matches: Dict[str, MatchResult],
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    matched_pattern: Any, qhandler: Optional[QuantizeHandler],
    is_qat: bool) -> Optional[Node]: ...


def maybe_insert_observers_before_graph_output(
    graph_output_node: Node, output_quantized_idxs: List[int],
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]],
    qconfig_map: Dict[str, QConfigAny], model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module], graph: Graph) -> None: ...


def maybe_propagate_dtype_for_node(
    node: Node, target_dtype: Union[torch.dtype, type],
    node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]], matches: Dict[str,
            MatchResult]) -> None: ...


def propagate_dtypes_for_known_nodes(
    graph: Graph, node_name_to_target_dtype: Dict[str, Dict[str,
                    Optional[Union[torch.dtype, type]]]], matches: Dict[str,
            MatchResult]) -> None: ...


def maybe_make_input_output_share_observers(
    node: Node, model: torch.nn.Module, modules: Dict[str,
            torch.nn.Module]) -> bool: ...


def remove_output_observer(
    node: Node, model: torch.nn.Module, modules: Dict[str,
            torch.nn.Module]): ...


def swap_custom_module_to_observed(
    node: Node, qconfig: QConfigAny, modules: Dict[str, torch.nn.Module],
    prepare_custom_config_dict: Dict[str, Any]): ...


def insert_observers_for_model(
    model: GraphModule, modules: Dict[str, torch.nn.Module],
    matches: Dict[str, MatchResult], qconfig_map: Dict[str, QConfigAny],
    graph: Graph, prepare_custom_config_dict: Dict[str, Any],
    equalization_config_map: Dict[str, Any], input_quantized_idxs: List[int],
    output_quantized_idxs: List[int], backend_config_dict: Optional[Dict[str,
                    Any]], observed_node_names: Set[
            str], is_qat: bool) -> Optional[Node]: ...


def run_prepare_fx_on_standalone_modules(
    model: torch.nn.Module, is_qat: bool, modules: Dict[str,
            torch.nn.Module], matches: Any,
    prepare_custom_config_dict: Dict[str, Any],
    backend_config_dict: Optional[Dict[str, Any]]) -> None: ...


def save_state(
    observed: GraphModule, qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
    prepare_custom_config_dict: Dict[str, Any],
    equalization_qconfig_map: Dict[str, Any], qconfig_dict: Dict[str,
            Dict[Any, Any]], is_qat: bool, observed_node_names: Set[
            str]) -> None: ...


def prepare(
    model: GraphModule, qconfig_dict: Any, is_qat: bool,
    node_name_to_scope: Dict[str, Tuple[str, type]],
    prepare_custom_config_dict: Optional[Dict[str, Any]] = ...,
    equalization_qconfig_dict: Optional[Dict[str, Any]] = ...,
    backend_config_dict: Optional[Dict[str, Any]] = ...,
    is_standalone_module: bool = ...) -> ObservedGraphModule: ...
