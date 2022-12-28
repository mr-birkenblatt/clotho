# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch.fx
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.node import map_aggregate as map_aggregate
from torch.fx.node import Node as Node


class TensorMetadata(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int]
    memory_format: Optional[torch.memory_format]
    is_quantized: bool
    qparams: Dict[str, Any]


class ShapeProp(torch.fx.Interpreter):
    def run_node(self, n: Node) -> Any: ...
    def propagate(self, *args): ...
