# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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
