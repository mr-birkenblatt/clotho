# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from torch.ao.quantization.quantize import add_observer_ as add_observer_
from torch.ao.quantization.quantize import (
    add_quant_dequant as add_quant_dequant,
)
from torch.ao.quantization.quantize import convert as convert
from torch.ao.quantization.quantize import (
    get_observer_dict as get_observer_dict,
)
from torch.ao.quantization.quantize import (
    get_unique_devices_ as get_unique_devices_,
)
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.ao.quantization.quantize import prepare as prepare
from torch.ao.quantization.quantize import prepare_qat as prepare_qat
from torch.ao.quantization.quantize import (
    propagate_qconfig_ as propagate_qconfig_,
)
from torch.ao.quantization.quantize import quantize as quantize
from torch.ao.quantization.quantize import quantize_dynamic as quantize_dynamic
from torch.ao.quantization.quantize import quantize_qat as quantize_qat
from torch.ao.quantization.quantize import (
    register_activation_post_process_hook as register_activation_post_process_hook,
)
from torch.ao.quantization.quantize import swap_module as swap_module
