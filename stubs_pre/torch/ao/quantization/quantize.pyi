# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.ao.quantization.qconfig import (
    activation_is_memoryless as activation_is_memoryless,
)
from torch.ao.quantization.qconfig import (
    add_module_to_qconfig_obs_ctr as add_module_to_qconfig_obs_ctr,
)
from torch.ao.quantization.qconfig import (
    default_dynamic_qconfig as default_dynamic_qconfig,
)
from torch.ao.quantization.qconfig import (
    float16_dynamic_qconfig as float16_dynamic_qconfig,
)
from torch.ao.quantization.qconfig import (
    float_qparams_weight_only_qconfig as float_qparams_weight_only_qconfig,
)
from torch.ao.quantization.qconfig import (
    float_qparams_weight_only_qconfig_4bit as float_qparams_weight_only_qconfig_4bit,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings as get_default_dynamic_quant_module_mappings,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_qat_module_mappings as get_default_qat_module_mappings,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_qconfig_propagation_list as get_default_qconfig_propagation_list,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_static_quant_module_mappings as get_default_static_quant_module_mappings,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_static_quant_reference_module_mappings as get_default_static_quant_reference_module_mappings,
)
from torch.ao.quantization.quantization_mappings import (
    no_observer_set as no_observer_set,
)
from torch.ao.quantization.stubs import DeQuantStub as DeQuantStub
from torch.ao.quantization.stubs import QuantWrapper as QuantWrapper
from torch.nn.utils.parametrize import (
    type_before_parametrizations as type_before_parametrizations,
)

from .utils import get_qparam_dict as get_qparam_dict
from .utils import (
    has_no_children_ignoring_parametrizations as has_no_children_ignoring_parametrizations,
)


def is_activation_post_process(module): ...


def propagate_qconfig_(
    module, qconfig_dict: Incomplete | None = ...,
    prepare_custom_config_dict: Incomplete | None = ...) -> None: ...


def register_activation_post_process_hook(
    module, pre_hook: bool = ...) -> None: ...


def add_observer_(
    module, qconfig_propagation_list: Incomplete | None = ...,
    non_leaf_module_list: Incomplete | None = ...,
    device: Incomplete | None = ...,
    custom_module_class_mapping: Incomplete | None = ...): ...


def get_unique_devices_(module): ...


def add_quant_dequant(module): ...


def prepare(
    model, inplace: bool = ..., allow_list: Incomplete | None = ...,
    observer_non_leaf_module_list: Incomplete | None = ...,
    prepare_custom_config_dict: Incomplete | None = ...): ...


def quantize(
    model, run_fn, run_args, mapping: Incomplete | None = ...,
    inplace: bool = ...): ...


def quantize_dynamic(
    model, qconfig_spec: Incomplete | None = ..., dtype=...,
    mapping: Incomplete | None = ..., inplace: bool = ...): ...


def prepare_qat(
    model, mapping: Incomplete | None = ..., inplace: bool = ...): ...


def quantize_qat(model, run_fn, run_args, inplace: bool = ...): ...


def convert(
    module, mapping: Incomplete | None = ..., inplace: bool = ...,
    remove_qconfig: bool = ..., is_reference: bool = ...,
    convert_custom_config_dict: Incomplete | None = ...): ...


def swap_module(mod, mapping, custom_module_class_mapping): ...


def get_observer_dict(mod, target_dict, prefix: str = ...): ...
