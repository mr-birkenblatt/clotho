# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, Dict, List, Tuple, Union

import torch.nn as nn

from ..quantization_types import Pattern as Pattern


def get_pattern_to_dtype_configs(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, List[Dict[
                        str, Any]]]: ...


def get_qat_module_classes(
    backend_config_dict: Dict[str, Any]) -> Tuple[type, ...]: ...


def get_fused_module_classes(
    backend_config_dict: Dict[str, Any]) -> Tuple[type, ...]: ...


def get_pattern_to_input_type_to_index(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Dict[str, int]]: ...


def get_root_module_to_quantized_reference_module(
    backend_config_dict: Dict[str, Any]) -> Dict[Callable, Callable]: ...


def get_fuser_method_mapping(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Union[
                nn.Sequential, Callable]]: ...


def get_module_to_qat_module(
    backend_config_dict: Dict[str, Any]) -> Dict[Callable, Callable]: ...


def get_fusion_pattern_to_root_node_getter(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]: ...


def get_fusion_pattern_to_extra_inputs_getter(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]: ...


def remove_boolean_dispatch_from_name(p) -> Any: ...


def pattern_to_human_readable(p) -> Any: ...


def entry_to_pretty_str(entry) -> str: ...
