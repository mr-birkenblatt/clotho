from typing import Callable, Dict, List, NamedTuple

from _typeshed import Incomplete

from ._compatibility import compatibility as compatibility
from ._symbolic_trace import symbolic_trace as symbolic_trace
from .graph import Graph as Graph
from .graph_module import GraphModule as GraphModule
from .node import Node as Node


class Match(NamedTuple):
    anchor: Node
    nodes_map: Dict[Node, Node]

class _SubgraphMatcher:
    pattern: Incomplete
    pattern_anchor: Incomplete
    nodes_map: Incomplete
    def __init__(self, pattern: Graph) -> None: ...
    def matches_subgraph_from_anchor(self, anchor: Node) -> bool: ...

def replace_pattern(gm: GraphModule, pattern: Callable, replacement: Callable) -> List[Match]: ...
