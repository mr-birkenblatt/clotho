# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.ao.quantization import QConfig as QConfig
from torch.ao.quantization.qconfig import (
    add_module_to_qconfig_obs_ctr as add_module_to_qconfig_obs_ctr,
)
from torch.ao.quantization.qconfig import qconfig_equals as qconfig_equals
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Graph as Graph

from ..qconfig_dict_utils import (
    get_object_type_qconfig as get_object_type_qconfig,
)
from ..qconfig_dict_utils import (
    maybe_adjust_qconfig_for_module_type_or_name as maybe_adjust_qconfig_for_module_type_or_name,
)
from ..utils import get_qconfig_dtypes as get_qconfig_dtypes


def maybe_adjust_qconfig_for_module_name_object_type_order(
    qconfig_dict: Any, cur_module_path: str, cur_object_type: Callable,
    cur_object_type_idx: int, fallback_qconfig: QConfigAny) -> QConfigAny: ...


def update_qconfig_for_fusion(
    model: GraphModule, qconfig_dict: Any) -> Any: ...


def generate_qconfig_map(
    root: torch.nn.Module, modules: Dict[str, torch.nn.Module],
    input_graph: Graph, qconfig_dict: Any, node_name_to_scope: Dict[str,
            Tuple[str, type]]) -> Dict[str, QConfigAny]: ...


def check_is_valid_config_dict(
    config_dict: Any, allowed_keys: Set[str], dict_name: str) -> None: ...


def check_is_valid_qconfig_dict(qconfig_dict: Any) -> None: ...


def check_is_valid_prepare_custom_config_dict(
    prepare_custom_config_dict: Optional[Dict[str, Any]] = ...) -> None: ...


def check_is_valid_convert_custom_config_dict(
    convert_custom_config_dict: Optional[Dict[str, Any]] = ...) -> None: ...


def check_is_valid_fuse_custom_config_dict(
    fuse_custom_config_dict: Optional[Dict[str, Any]] = ...) -> None: ...


def compare_prepare_convert_qconfig_dict(
    prepare_qconfig_dict: Dict[str, Dict[Any, Any]],
    convert_qconfig_dict: Dict[str, Dict[Any, Any]]) -> None: ...


def is_qconfig_supported_by_dtype_configs(
    qconfig: QConfig, dtype_configs: List[Dict[str, Any]]): ...


def get_standalone_module_configs(
    module_name: str, module_type: Callable, custom_config_dict: Dict[str,
            Any]): ...
