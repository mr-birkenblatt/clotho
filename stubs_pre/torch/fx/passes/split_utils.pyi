# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Dict, List, Optional

import torch.fx
import torch.nn as nn
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.graph import map_arg as map_arg

from .tools_common import NodeList as NodeList
from .tools_common import NodeSet as NodeSet


def getattr_recursive(obj, name): ...


def setattr_recursive(obj, attr, value) -> None: ...


class Component:
    graph: torch.fx.Graph
    order: int
    name: str
    input_placeholders: List
    orig_inputs: List
    orig_outputs: List
    getattr_maps: Dict[torch.fx.Node, torch.fx.Node]
    constructor_args: List[str]
    gm: Optional[torch.fx.GraphModule]

    def __init__(
        self, graph, order, name, input_placeholders, orig_inputs,
        orig_outputs, getattr_maps, constructor_args, gm) -> None: ...


class HolderModule(nn.Module):
    def __init__(self, d) -> None: ...


def split_by_tags(
    gm: torch.fx.GraphModule, tags: List[str]) -> torch.fx.GraphModule: ...
