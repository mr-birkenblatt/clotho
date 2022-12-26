from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type

from _typeshed import Incomplete


Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[List, Context], PyTree]

class NodeDef(NamedTuple):
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc

SUPPORTED_NODES: Dict[Type[Any], NodeDef]

class TreeSpec:
    type: Incomplete
    context: Incomplete
    children_specs: Incomplete
    num_leaves: Incomplete
    def __init__(self, typ: Any, context: Context, children_specs: List['TreeSpec']) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...

class LeafSpec(TreeSpec):
    num_leaves: int
    def __init__(self) -> None: ...

def tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]: ...
def tree_unflatten(values: List[Any], spec: TreeSpec) -> PyTree: ...
def tree_map(fn: Any, pytree: PyTree) -> PyTree: ...