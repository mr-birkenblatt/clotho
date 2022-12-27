# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Dict, Optional

import torch
from _typeshed import Incomplete
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.graph_module import GraphModule as GraphModule


class Partition:
    name: Incomplete
    submod_name: Incomplete
    node_names: Incomplete
    inputs: Incomplete
    outputs: Incomplete
    partitions_dependent_on: Incomplete
    partition_dependents: Incomplete
    graph: Incomplete
    environment: Incomplete
    targets: Incomplete
    def __init__(self, name: str) -> None: ...


def split_module(
    m: GraphModule, root_m: torch.nn.Module,
    split_callback: Callable[[torch.fx.node.Node], int],
    qualname_map: Optional[Dict[str, str]] = ...): ...
