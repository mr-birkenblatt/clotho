# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Dict, List, Mapping, Set, Union

import torch.fx
from _typeshed import Incomplete
from torch.fx._compatibility import compatibility as compatibility


Tensors: Incomplete
TensorOrTensors: Incomplete
NodeList = List[torch.fx.Node]
NodeSet = Set[torch.fx.Node]
Names = List[str]
CALLABLE_NODE_OPS: Incomplete


def get_acc_ops_name(k): ...


def get_node_target(
    submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> str: ...


def is_node_output_tensor(node: torch.fx.Node) -> bool: ...


class FxNetAccFusionsFinder:
    module: Incomplete
    nodes: Incomplete
    acc_nodes: Incomplete

    def __init__(
        self, module: torch.fx.GraphModule, acc_nodes: NodeSet) -> None: ...

    class FusionGroup:
        top_node_idx: int
        nodes: NodeSet
        inputs: NodeSet
        nodes_need_process: NodeSet
        def add_node(self, node) -> None: ...

        def __init__(
            self, top_node_idx, nodes, inputs, nodes_need_process) -> None: ...

    def recursive_add_node(
        self, fusion_group: FxNetAccFusionsFinder.FusionGroup,
        inputs: Union[NodeSet, NodeList]): ...

    def __call__(self) -> Dict[torch.fx.Node, NodeSet]: ...


def legalize_graph(gm: torch.fx.GraphModule): ...
