# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Tuple, Union

import torch
from _typeshed import Incomplete
from torch.ao.quantization.quant_type import (
    quant_type_to_str as quant_type_to_str,
)
from torch.ao.quantization.quant_type import QuantType as QuantType
from torch.nn.utils.parametrize import is_parametrized as is_parametrized


Pattern = Union[Callable, Tuple[Callable, Callable], Tuple[Callable,
                    Tuple[Callable, Callable]], Any]


class MatchAllNode:
    ...


module_type_list: Incomplete
func_list: Incomplete
method_list: Incomplete


def check_node(node, modules): ...


def get_combined_dict(default_dict, additional_dict): ...


def is_per_tensor(qscheme): ...


def is_per_channel(qscheme): ...


def getattr_from_fqn(obj: Any, fqn: str) -> Any: ...


def get_qparam_dict(observer_or_fake_quant): ...


def get_swapped_custom_module_class(
    custom_module, custom_module_class_mapping, qconfig): ...


def activation_dtype(qconfig): ...


def weight_dtype(qconfig): ...


def activation_is_statically_quantized(qconfig): ...


def activation_is_dynamically_quantized(qconfig): ...


def activation_is_int8_quantized(qconfig): ...


def activation_is_int32_quantized(qconfig): ...


def weight_is_quantized(qconfig): ...


def weight_is_statically_quantized(qconfig): ...


def op_is_int8_dynamically_quantized(qconfig) -> bool: ...


def get_qconfig_dtypes(qconfig): ...


def get_quant_type(qconfig): ...


def check_min_max_valid(
    min_val: torch.Tensor, max_val: torch.Tensor) -> bool: ...


def calculate_qmin_qmax(
    quant_min: int, quant_max: int, has_customized_qrange: bool,
    dtype: torch.dtype, reduce_range: bool) -> Tuple[int, int]: ...


def has_no_children_ignoring_parametrizations(module): ...
