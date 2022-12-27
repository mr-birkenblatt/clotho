# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, Dict

from torch.ao.quantization.quantization_types import Pattern, QuantizerCls


def get_pattern_to_quantize_handlers(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, QuantizerCls]: ...


def get_fusion_pattern_to_fuse_handler_cls(
    backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]: ...


def get_native_quant_patterns(
    additional_quant_patterns: Dict[Pattern, QuantizerCls] = ...) -> Dict[
        Pattern, QuantizerCls]: ...
