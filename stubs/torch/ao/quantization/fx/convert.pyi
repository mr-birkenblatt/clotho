# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Optional, Set, Tuple

import torch
from torch.ao.quantization.backend_config import (
    get_native_backend_config_dict as get_native_backend_config_dict,
)
from torch.ao.quantization.backend_config.utils import (
    get_fused_module_classes as get_fused_module_classes,
)
from torch.ao.quantization.backend_config.utils import (
    get_pattern_to_dtype_configs as get_pattern_to_dtype_configs,
)
from torch.ao.quantization.backend_config.utils import (
    get_qat_module_classes as get_qat_module_classes,
)
from torch.ao.quantization.backend_config.utils import (
    get_root_module_to_quantized_reference_module as get_root_module_to_quantized_reference_module,
)
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Argument as Argument
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node

from ..qconfig import qconfig_equals as qconfig_equals
from ..qconfig import QConfigAny as QConfigAny
from ..qconfig_dict_utils import (
    convert_dict_to_ordered_dict as convert_dict_to_ordered_dict,
)
from ..qconfig_dict_utils import (
    update_qconfig_for_qat as update_qconfig_for_qat,
)
from ..utils import (
    activation_is_statically_quantized as activation_is_statically_quantized,
)
from ..utils import get_qparam_dict as get_qparam_dict
from ..utils import (
    get_swapped_custom_module_class as get_swapped_custom_module_class,
)
from ..utils import weight_is_quantized as weight_is_quantized
from ._equalize import convert_eq_obs as convert_eq_obs
from ._equalize import (
    update_obs_for_equalization as update_obs_for_equalization,
)
from .graph_module import is_observed_module as is_observed_module
from .graph_module import (
    is_observed_standalone_module as is_observed_standalone_module,
)
from .graph_module import QuantizedGraphModule as QuantizedGraphModule
from .lower_to_fbgemm import lower_to_fbgemm as lower_to_fbgemm
from .qconfig_utils import (
    compare_prepare_convert_qconfig_dict as compare_prepare_convert_qconfig_dict,
)
from .qconfig_utils import generate_qconfig_map as generate_qconfig_map
from .qconfig_utils import (
    is_qconfig_supported_by_dtype_configs as is_qconfig_supported_by_dtype_configs,
)
from .qconfig_utils import (
    update_qconfig_for_fusion as update_qconfig_for_fusion,
)
from .utils import collect_producer_nodes as collect_producer_nodes
from .utils import create_getattr_from_value as create_getattr_from_value
from .utils import get_custom_module_class_keys as get_custom_module_class_keys
from .utils import get_quantize_node_info as get_quantize_node_info
from .utils import (
    graph_module_from_producer_nodes as graph_module_from_producer_nodes,
)
from .utils import WEIGHT_INDEX_DICT as WEIGHT_INDEX_DICT


def restore_state(
    observed: torch.nn.Module) -> Tuple[Dict[str, Tuple[str, type]],
        Dict[str, Any], Set[str]]: ...


def has_none_qconfig(
    node: Argument, qconfig_map: Dict[str, QConfigAny]) -> bool: ...


def run_weight_observers(observed: GraphModule) -> None: ...


def duplicate_quantize_dynamic_node(
    quantized: QuantizedGraphModule) -> QuantizedGraphModule: ...


def duplicate_dequantize_node(
    quantized: QuantizedGraphModule) -> QuantizedGraphModule: ...


def remove_extra_dequantize(
    quantized: QuantizedGraphModule) -> QuantizedGraphModule: ...


def remove_quant_dequant_pairs(
    quantized: QuantizedGraphModule) -> QuantizedGraphModule: ...


def maybe_recursive_remove_dequantize(arg: Any, node: Node, graph: Graph): ...


def get_module_path_and_prefix(
    obs_node: Node, node_name_to_scope: Dict[str, Tuple[str, type]],
        qconfig_map: Dict[str, QConfigAny]): ...


def insert_dequantize_node(node: Node, graph: Graph): ...


def maybe_get_observer_for_node(
    node: Node, modules: Dict[str,
            torch.nn.Module]) -> Optional[torch.nn.Module]: ...


def convert_standalone_module(
    node: Node, modules: Dict[str, torch.nn.Module],
        model: torch.fx.GraphModule, is_reference: bool,
        backend_config_dict: Optional[Dict[str, Any]]): ...


def convert_weighted_module(
    node: Node, modules: Dict[str, torch.nn.Module],
        observed_node_names: Set[str], qconfig_map: Dict[str, QConfigAny],
        backend_config_dict: Dict[str, Any]): ...


def convert_custom_module(
    node: Node, graph: Graph, modules: Dict[str, torch.nn.Module],
        custom_module_class_mapping: Dict[Callable, Callable],
        statically_quantized_custom_module_nodes: Set[Node]): ...


def convert(
    model: GraphModule, is_reference: bool = ...,
        convert_custom_config_dict: Dict[str, Any] = ...,
        is_standalone_module: bool = ..., _remove_qconfig_flag: bool = ...,
        convert_qconfig_dict: Dict[str, Any] = ...,
        backend_config_dict: Optional[Dict[str,
                Any]] = ...) -> torch.nn.Module: ...
