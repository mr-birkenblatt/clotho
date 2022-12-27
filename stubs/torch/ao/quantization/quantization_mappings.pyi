# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Optional, Set, Union

from torch import nn as nn
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant as default_fixed_qparams_range_0to1_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_neg1to1_fake_quant as default_fixed_qparams_range_neg1to1_fake_quant,
)
from torch.ao.quantization.stubs import DeQuantStub as DeQuantStub
from torch.ao.quantization.stubs import QuantStub as QuantStub
from torch.ao.quantization.utils import get_combined_dict as get_combined_dict
from torch.nn.utils.parametrize import (
    type_before_parametrizations as type_before_parametrizations,
)


DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any]
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any]
DEFAULT_QAT_MODULE_MAPPINGS: Dict[Callable, Any]
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: Dict[Callable, Any]
DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS: Dict[Union[Callable, str],
        Callable]
DEFAULT_MODULE_TO_ACT_POST_PROCESS: Dict[Callable, Callable]
DEFAULT_STATIC_SPARSE_QUANT_MODULE_MAPPINGS: Dict[Callable, Any]
DEFAULT_DYNAMIC_SPARSE_QUANT_MODULE_MAPPINGS: Dict[Callable, Any]


def no_observer_set() -> Set[Any]: ...


def get_default_static_quant_module_mappings() -> Dict[Callable, Any]: ...


def get_default_static_quant_reference_module_mappings(
    ) -> Dict[Callable, Any]: ...


def get_embedding_static_quant_module_mappings() -> Dict[Callable, Any]: ...


def get_default_static_sparse_quant_module_mappings(
    ) -> Dict[Callable, Any]: ...


def get_static_quant_module_class(
    float_module_class: Callable,
        additional_static_quant_mapping: Optional[Dict[Callable, Any]] = ...,
        is_reference: bool = ...) -> Any: ...


def get_dynamic_quant_module_class(
    float_module_class: Callable,
        additional_dynamic_quant_mapping: Optional[Dict[Callable,
                Any]] = ...) -> Any: ...


def get_default_qat_module_mappings() -> Dict[Callable, Any]: ...


def get_embedding_qat_module_mappings() -> Dict[Callable, Any]: ...


def get_default_dynamic_quant_module_mappings() -> Dict[Callable, Any]: ...


def get_default_dynamic_sparse_quant_module_mappings(
    ) -> Dict[Callable, Any]: ...


def get_default_qconfig_propagation_list() -> Set[Callable]: ...


def get_default_compare_output_module_list() -> Set[Callable]: ...


def get_default_float_to_quantized_operator_mappings(
    ) -> Dict[Union[Callable, str], Callable]: ...


def get_quantized_operator(float_op: Union[Callable, str]) -> Callable: ...
