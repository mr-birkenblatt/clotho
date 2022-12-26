from typing import List

import torch
from torch.fx._symbolic_trace import symbolic_trace as symbolic_trace
from torch.fx.node import Node as Node
from torch.fx.passes.tools_common import legalize_graph as legalize_graph


def split_result_tensors(result: torch.Tensor, inputs: List[torch.Tensor]) -> List[torch.Tensor]: ...
def may_depend_on(a: Node, b: Node, search_depth: int = ...): ...
def are_nodes_independent(nodes: List[Node]): ...
def merge_matmul(in_mod: torch.nn.Module): ...
