from typing import Callable, Dict, Sequence, Union

import torch._refs
from _typeshed import Incomplete


decomposition_table: Dict[torch._ops.OpOverload, Callable]

def register_decomposition(aten_op, registry: Incomplete | None = ..., *, disable_meta: bool = ...): ...
def get_decompositions(aten_ops: Sequence[Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]]) -> Dict[torch._ops.OpOverload, Callable]: ...