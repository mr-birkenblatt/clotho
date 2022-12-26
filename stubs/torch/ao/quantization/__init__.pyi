from .fake_quantize import *
from .fuse_modules import fuse_modules as fuse_modules
from .fuse_modules import fuse_modules_qat as fuse_modules_qat
from .fuser_method_mappings import *
from .observer import *
from .qconfig import *
from .quant_type import *
from .quantization_mappings import *
from .quantize import *
from .quantize_jit import *
from .stubs import *


def default_eval_fn(model, calib_data) -> None: ...
