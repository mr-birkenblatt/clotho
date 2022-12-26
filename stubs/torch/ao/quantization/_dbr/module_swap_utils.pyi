from torch.ao.quantization import swap_module as swap_module
from torch.ao.quantization.quantization_mappings import (
    DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS as DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
)

from ..utils import (
    activation_is_int8_quantized as activation_is_int8_quantized,
)
from ..utils import (
    activation_is_int32_quantized as activation_is_int32_quantized,
)
from ..utils import (
    op_is_int8_dynamically_quantized as op_is_int8_dynamically_quantized,
)
