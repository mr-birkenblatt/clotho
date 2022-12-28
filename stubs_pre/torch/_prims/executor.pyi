# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable

from torch._C._nvfuser import Fusion as Fusion
from torch._C._nvfuser import FusionDefinition as FusionDefinition
from torch._prims.context import PrimContext as PrimContext
from torch._prims.utils import getnvFuserDtype as getnvFuserDtype
from torch._prims.utils import TensorMeta as TensorMeta
from torch.fx import GraphModule as GraphModule


def execute(ctx: PrimContext, *args, executor: str = ..., **kwargs): ...


def make_traced(fn: Callable): ...
