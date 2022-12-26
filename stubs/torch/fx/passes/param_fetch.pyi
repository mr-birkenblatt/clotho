from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.graph_module import GraphModule as GraphModule


def default_matching(name: str, target_version: int) -> str: ...

module_fetch_book: Dict[Type, Tuple[int, List[str], Callable[[str, int], str]]]

def extract_attrs_for_lowering(mod: nn.Module) -> Dict[str, Any]: ...
def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None: ...
