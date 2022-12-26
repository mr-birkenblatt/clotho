import enum
from typing import Any, Callable, Dict, List, Union

from _typeshed import Incomplete
from torch.fx.graph import Node as Node


class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT: str
    NODE_OUTPUT: str
    NODE_INPUT: str

NSSubgraph: Incomplete
NSSingleResultType = Dict[str, Any]
NSResultsType = Dict[str, Dict[str, Dict[str, List[NSSingleResultType]]]]
NSNodeTargetType = Union[Callable, str]
