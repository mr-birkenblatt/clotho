# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


from torch.ao.quantization.quantization_mappings import
        DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS as
        DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS as
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
        DEFAULT_STATIC_QUANT_MODULE_MAPPINGS as
        DEFAULT_STATIC_QUANT_MODULE_MAPPINGS
from typing import Callable


toq: Incomplete
fp32_to_int8_fun_mapping: Incomplete
functions_supported_by_quantization: Incomplete
module_types_supported_by_quantization: Incomplete
module_types_supported_by_quantization_preserves_dtype: Incomplete
functions_supported_by_quantization_preserves_dtype: Incomplete
add_and_mul_ops: Incomplete
known_module_fusion_patterns: Incomplete
known_function_fusion_patterns_and_replacements: Incomplete
binary_related_ops: Incomplete
conv_ops: Incomplete
conv_prepack_fns: Incomplete
a_related_to_b: Incomplete


def ops_are_related(
    cur_op: Callable, expected_op_type: Callable,
    type_is_module: bool) -> bool: ...
