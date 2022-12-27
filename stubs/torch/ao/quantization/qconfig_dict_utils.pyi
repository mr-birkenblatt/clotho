# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Union

from torch.ao.quantization.qconfig import QConfigAny as QConfigAny

from .quantization_mappings import (
    get_default_qat_module_mappings as get_default_qat_module_mappings,
)
from .utils import get_combined_dict as get_combined_dict


def get_object_type_qconfig(
    qconfig_dict: Any, object_type: Union[Callable, str],
    fallback_qconfig: QConfigAny) -> QConfigAny: ...


def get_module_name_regex_qconfig(
    qconfig_dict, module_name, fallback_qconfig): ...


def get_module_name_qconfig(qconfig_dict, module_name, fallback_qconfig): ...


def maybe_adjust_qconfig_for_module_type_or_name(
    qconfig_dict, module_type, module_name, global_qconfig): ...


def get_flattened_qconfig_dict(qconfig_dict): ...


def convert_dict_to_ordered_dict(
    qconfig_dict: Any) -> Dict[str, Dict[Any, Any]]: ...


def update_qconfig_for_qat(
    qconfig_dict: Any, additional_qat_module_mapping: Dict[Callable,
            Callable]) -> Any: ...
