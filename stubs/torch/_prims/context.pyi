# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Sequence

import torch.overrides
from _typeshed import Incomplete
from torch._prims.utils import TensorMeta as TensorMeta
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node


class PrimContext(torch.overrides.TorchFunctionMode):
    graph: Incomplete
    def __init__(self) -> None: ...
    def placeholder(self, a: Any): ...
    def output(self, tm: TensorMeta): ...

    def __torch_function__(
        self, func: Callable, types: Sequence, args: Sequence[Any] = ...,
        kwargs: Dict = ...): ...
