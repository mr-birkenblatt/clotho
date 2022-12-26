from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch
from _typeshed import Incomplete
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.node import map_arg as map_arg
from torch.fx.passes.graph_manipulation import (
    get_size_of_node as get_size_of_node,
)

from .graph_drawer import FxGraphDrawer as FxGraphDrawer
from .operator_support import get_node_target as get_node_target
from .operator_support import OperatorSupportBase as OperatorSupportBase
from .shape_prop import ShapeProp as ShapeProp
from .split_utils import split_by_tags as split_by_tags
from .tools_common import CALLABLE_NODE_OPS as CALLABLE_NODE_OPS
from .tools_common import FxNetAccFusionsFinder as FxNetAccFusionsFinder
from .tools_common import is_node_output_tensor as is_node_output_tensor
from .tools_common import NodeList as NodeList
from .tools_common import NodeSet as NodeSet
from .tools_common import Tensors as Tensors


class _SplitterSettingBase:
    min_acc_module_size: Incomplete
    skip_fusion: Incomplete
    allow_non_tensor: Incomplete
    def __init__(self) -> None: ...

class FxNetAccNodesFinder:
    module: Incomplete
    operator_support: Incomplete
    allow_non_tensor: Incomplete
    def __init__(self, module: torch.fx.GraphModule, operator_support: OperatorSupportBase, allow_non_tensor: bool) -> None: ...
    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList): ...
    def reduce_acc_nodes_non_tensor_input(self) -> None: ...
    def reduce_acc_nodes_non_tensor_output(self) -> None: ...
    acc_nodes: Incomplete
    def __call__(self) -> NodeSet: ...

class FxNetSplitterInternalError(Exception): ...

class Subgraph:
    is_acc: bool
    nodes: NodeList
    def __init__(self, is_acc, nodes) -> None: ...

class SplitResult(NamedTuple):
    split_module: torch.fx.GraphModule
    submodule_inputs: Dict[str, Any]
    non_acc_submodule_prefix: str

def generate_inputs_for_submodules(model: torch.nn.Module, inputs: Sequence[Any], target_submodules: Iterable[str]) -> Dict[str, Any]: ...

class _SplitterBase:
    PCIe_BW: Incomplete
    module: Incomplete
    settings: Incomplete
    operator_support: Incomplete
    sample_input: Incomplete
    acc_nodes: Incomplete
    fusions: Incomplete
    deps: Incomplete
    non_acc_submodule_name: Incomplete
    def __init__(self, module: torch.fx.GraphModule, sample_input: Sequence[Any], operator_support: OperatorSupportBase, settings: _SplitterSettingBase, non_acc_submodule_name: str = ...) -> None: ...
    def find_deps(self) -> Dict[torch.fx.Node, NodeSet]: ...
    def update_deps_for_fusions(self) -> None: ...
    def node_support_preview(self, dump_graph: bool = ...): ...
    def split_preview(self, dump_graph: bool = ...): ...
    def find_reverse_deps(self, tag_id: Optional[int] = ...) -> Dict[torch.fx.Node, NodeSet]: ...
    def update_reverse_deps_for_fusions(self, deps: Dict[torch.fx.Node, NodeSet]): ...
    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet: ...
    def extend_acc_subgraph(self, tag: str): ...
    def starter_nodes(self) -> Tuple[NodeSet, NodeSet]: ...
    def put_nodes_into_subgraphs(self) -> List[Subgraph]: ...
    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]: ...
    tags: Incomplete
    def tag(self, subgraphs: List[Subgraph]): ...
    def split(self, remove_tag: bool = ...) -> torch.fx.GraphModule: ...
    def __call__(self) -> torch.fx.GraphModule: ...
    def generate_split_results(self) -> SplitResult: ...
