# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .fake_quantize import *
from .fuse_modules import fuse_modules as fuse_modules
from .fuser_method_mappings import *
from .observer import *
from .qconfig import *
from .quant_type import *
from .quantization_mappings import *
from .quantize import *
from .quantize_jit import *
from .stubs import *


        fuse_modules_qat as fuse_modules_qat


def default_eval_fn(model, calib_data) -> None: ...
