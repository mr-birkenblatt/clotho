# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.qconfig import (
    add_module_to_qconfig_obs_ctr as add_module_to_qconfig_obs_ctr,
)
from torch.ao.quantization.qconfig import (
    assert_valid_qconfig as assert_valid_qconfig,
)
from torch.ao.quantization.qconfig import (
    default_activation_only_qconfig as default_activation_only_qconfig,
)
from torch.ao.quantization.qconfig import (
    default_debug_qconfig as default_debug_qconfig,
)
from torch.ao.quantization.qconfig import (
    default_dynamic_qconfig as default_dynamic_qconfig,
)
from torch.ao.quantization.qconfig import (
    default_per_channel_qconfig as default_per_channel_qconfig,
)
from torch.ao.quantization.qconfig import (
    default_qat_qconfig as default_qat_qconfig,
)
from torch.ao.quantization.qconfig import (
    default_qat_qconfig_v2 as default_qat_qconfig_v2,
)
from torch.ao.quantization.qconfig import default_qconfig as default_qconfig
from torch.ao.quantization.qconfig import (
    default_weight_only_qconfig as default_weight_only_qconfig,
)
from torch.ao.quantization.qconfig import (
    float16_dynamic_qconfig as float16_dynamic_qconfig,
)
from torch.ao.quantization.qconfig import (
    float16_static_qconfig as float16_static_qconfig,
)
from torch.ao.quantization.qconfig import (
    float_qparams_weight_only_qconfig as float_qparams_weight_only_qconfig,
)
from torch.ao.quantization.qconfig import (
    get_default_qat_qconfig as get_default_qat_qconfig,
)
from torch.ao.quantization.qconfig import (
    get_default_qconfig as get_default_qconfig,
)
from torch.ao.quantization.qconfig import (
    per_channel_dynamic_qconfig as per_channel_dynamic_qconfig,
)
from torch.ao.quantization.qconfig import QConfig as QConfig
from torch.ao.quantization.qconfig import qconfig_equals as qconfig_equals
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny
from torch.ao.quantization.qconfig import QConfigDynamic as QConfigDynamic
