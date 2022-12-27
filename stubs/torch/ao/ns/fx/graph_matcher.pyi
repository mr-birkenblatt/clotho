# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import enum
from typing import Dict, Optional, Set, Tuple

from _typeshed import Incomplete
from torch.ao.quantization import FakeQuantizeBase as FakeQuantizeBase
from torch.ao.quantization import ObserverBase as ObserverBase
from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node

from .mappings import (
    get_base_name_to_sets_of_related_ops as get_base_name_to_sets_of_related_ops,
)
from .mappings import get_unmatchable_types_map as get_unmatchable_types_map
from .ns_types import NSNodeTargetType as NSNodeTargetType
from .ns_types import NSSubgraph as NSSubgraph
from .pattern_utils import (
    end_node_matches_reversed_fusion as end_node_matches_reversed_fusion,
)
from .pattern_utils import get_reversed_fusions as get_reversed_fusions
from .pattern_utils import get_type_a_related_to_b as get_type_a_related_to_b


toq: Incomplete


class _NSGraphMatchableSubgraphsIterator:
    gm: Incomplete
    non_matchable_functions: Incomplete
    non_matchable_modules: Incomplete
    non_matchable_methods: Incomplete
    seen_nodes: Incomplete
    stack: Incomplete

    def __init__(
        self, gm: GraphModule,
        non_matchable_functions: Set[NSNodeTargetType],
        non_matchable_modules: Set[NSNodeTargetType],
        non_matchable_methods: Set[NSNodeTargetType]) -> None: ...

    def __iter__(self): ...
    def __next__(self) -> NSSubgraph: ...


class GraphMatchingException(Exception):
    ...


class SubgraphTypeRelationship(enum.Enum):
    EQUAL: Incomplete
    EQUAL_BUT_UKNOWN: Incomplete
    RELATED_BUT_NOT_EQUAL: Incomplete
    NOT_RELATED: Incomplete


def get_matching_subgraph_pairs(
    gm_a: GraphModule, gm_b: GraphModule,
    base_name_to_sets_of_related_ops: Optional[Dict[str,
                    Set[NSNodeTargetType]]] = ...,
    unmatchable_types_map: Optional[Dict[str,
                    Set[NSNodeTargetType]]] = ...) -> Dict[str, Tuple[
                NSSubgraph, NSSubgraph]]: ...
