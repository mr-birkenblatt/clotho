from typing import Any, Callable, Dict, List, Type

from torch.utils._pytree import LeafSpec as LeafSpec
from torch.utils._pytree import PyTree as PyTree
from torch.utils._pytree import TreeSpec as TreeSpec


FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]
SUPPORTED_NODES: Dict[Type[Any], Any]

def register_pytree_flatten_spec(typ: Any, flatten_fn_spec: FlattenFuncSpec) -> None: ...
def tree_flatten_spec(pytree: PyTree, spec: TreeSpec) -> List[Any]: ...
