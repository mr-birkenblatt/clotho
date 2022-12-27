# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.ao.quantization import FakeQuantizeBase as FakeQuantizeBase

from .ns_types import NSNodeTargetType as NSNodeTargetType


        ObserverBase as ObserverBase
from torch.ao.quantization.fx.backend_config_utils import
        get_native_quant_patterns as get_native_quant_patterns
from typing import Any, Callable, Dict, List, Set, Tuple, Union

from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node


toq: Incomplete


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: Dict[str,
    Set[NSNodeTargetType]]) -> Set[Tuple[NSNodeTargetType,
        NSNodeTargetType]]: ...


NSFusionElType = Union[Callable, str, Tuple[str, Any]]
NSFusionType = Union[Tuple[NSFusionElType, NSFusionElType],
        Tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType]]


def get_reversed_fusions() -> List[Tuple[NSFusionType, int]]: ...


def end_node_matches_reversed_fusion(
    end_node: Node, reversed_fusion: NSFusionType, gm: GraphModule,
    seen_nodes: Set[Node]) -> bool: ...
