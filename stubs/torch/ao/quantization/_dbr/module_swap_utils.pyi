# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from ..utils import (
    activation_is_int32_quantized as activation_is_int32_quantized,
)


        activation_is_int8_quantized as activation_is_int8_quantized,
        op_is_int8_dynamically_quantized as op_is_int8_dynamically_quantized
from torch.ao.quantization import swap_module as swap_module
from torch.ao.quantization.quantization_mappings import (
    DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS as DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
)
