from typing import Callable

from torch._C._nvfuser import Fusion as Fusion
from torch._C._nvfuser import FusionDefinition as FusionDefinition
from torch._prims.context import PrimContext as PrimContext
from torch._prims.utils import getnvFuserDtype as getnvFuserDtype
from torch._prims.utils import TensorMeta as TensorMeta
from torch.fx import GraphModule as GraphModule


def execute(ctx: PrimContext, *args, executor: str = ..., **kwargs): ...
def make_traced(fn: Callable): ...
