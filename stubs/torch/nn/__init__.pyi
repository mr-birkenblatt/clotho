from . import functional as functional
from . import init as init
from . import utils as utils
from .modules import *
from .parallel import DataParallel as DataParallel
from .parameter import Parameter as Parameter
from .parameter import UninitializedBuffer as UninitializedBuffer
from .parameter import UninitializedParameter as UninitializedParameter


def factory_kwargs(kwargs): ...
