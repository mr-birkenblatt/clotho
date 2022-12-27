# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch

from ._dbr.auto_trace import add_auto_convert as add_auto_convert


        add_auto_observation as add_auto_observation
from ._dbr.fusion import get_module_fusion_fqns as get_module_fusion_fqns
from ._dbr.qconfig_dict_utils import as, normalize_object_types


        normalize_object_types
from .qconfig_dict_utils import as, convert_dict_to_ordered_dict


        convert_dict_to_ordered_dict,
        get_flattened_qconfig_dict as get_flattened_qconfig_dict
from _typeshed import Incomplete


from torch.ao.quantization.quantization_mappings import
        get_default_dynamic_quant_module_mappings as
        get_default_dynamic_quant_module_mappings,
        get_default_static_quant_module_mappings as
        get_default_static_quant_module_mappings


def prepare(
    model, qconfig_dict, example_inputs, inplace: bool = ...,
    allow_list: Incomplete | None = ...,
    observer_non_leaf_module_list: Incomplete | None = ...,
    prepare_custom_config_dict: Incomplete | None = ...,
    fuse_modules: bool = ...): ...


def convert(model: torch.nn.Module) -> torch.nn.Module: ...
