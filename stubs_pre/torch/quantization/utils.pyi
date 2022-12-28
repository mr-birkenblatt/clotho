# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from torch.ao.quantization.utils import activation_dtype as activation_dtype
from torch.ao.quantization.utils import (
    activation_is_int8_quantized as activation_is_int8_quantized,
)
from torch.ao.quantization.utils import (
    activation_is_statically_quantized as activation_is_statically_quantized,
)
from torch.ao.quantization.utils import (
    calculate_qmin_qmax as calculate_qmin_qmax,
)
from torch.ao.quantization.utils import (
    check_min_max_valid as check_min_max_valid,
)
from torch.ao.quantization.utils import get_combined_dict as get_combined_dict
from torch.ao.quantization.utils import (
    get_qconfig_dtypes as get_qconfig_dtypes,
)
from torch.ao.quantization.utils import get_qparam_dict as get_qparam_dict
from torch.ao.quantization.utils import get_quant_type as get_quant_type
from torch.ao.quantization.utils import (
    get_swapped_custom_module_class as get_swapped_custom_module_class,
)
from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.ao.quantization.utils import is_per_channel as is_per_channel
from torch.ao.quantization.utils import is_per_tensor as is_per_tensor
from torch.ao.quantization.utils import weight_dtype as weight_dtype
from torch.ao.quantization.utils import (
    weight_is_quantized as weight_is_quantized,
)
from torch.ao.quantization.utils import (
    weight_is_statically_quantized as weight_is_statically_quantized,
)
