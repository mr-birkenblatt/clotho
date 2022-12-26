from typing import Callable, Optional, Union

import torch.fx
from _typeshed import Incomplete
from torch.fx.node import map_arg as map_arg
from torch.fx.passes.split_module import split_module as split_module


class FoldedGraphModule(torch.fx.GraphModule):
    const_subgraph_module: Incomplete
    has_folding_been_run: bool
    fx_const_folded_attrs_name: Incomplete
    def __init__(self, root: torch.nn.Module, graph: torch.fx.Graph, const_subgraph: Optional[torch.fx.Graph] = ..., fx_const_folded_attrs_name: str = ...) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def run_folding(self) -> None: ...

def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str: ...
def split_const_subgraphs(module: Union[torch.nn.Module, torch.fx.GraphModule], skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = ...) -> FoldedGraphModule: ...