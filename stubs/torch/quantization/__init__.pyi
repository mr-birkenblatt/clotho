from _typeshed import Incomplete

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


def default_eval_fn(model, calib_data) -> None: ...

_all__: Incomplete
