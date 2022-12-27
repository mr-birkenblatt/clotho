# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.quantize_jit import (
    convert_dynamic_jit as convert_dynamic_jit,
)
from torch.ao.quantization.quantize_jit import convert_jit as convert_jit
from torch.ao.quantization.quantize_jit import (
    fuse_conv_bn_jit as fuse_conv_bn_jit,
)
from torch.ao.quantization.quantize_jit import (
    prepare_dynamic_jit as prepare_dynamic_jit,
)
from torch.ao.quantization.quantize_jit import prepare_jit as prepare_jit
from torch.ao.quantization.quantize_jit import (
    quantize_dynamic_jit as quantize_dynamic_jit,
)
from torch.ao.quantization.quantize_jit import quantize_jit as quantize_jit
from torch.ao.quantization.quantize_jit import script_qconfig as script_qconfig
from torch.ao.quantization.quantize_jit import (
    script_qconfig_dict as script_qconfig_dict,
)
