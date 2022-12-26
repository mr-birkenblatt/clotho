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
