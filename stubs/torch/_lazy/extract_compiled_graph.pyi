from typing import Any, Callable, Dict, List

from _typeshed import Incomplete
from torch import fx as fx
from torch._lazy import computation as computation
from torch._lazy.tensor_factory_functions import (
    tensor_factory_functions as tensor_factory_functions,
)


debug: Incomplete

class GraphInputMatcher:
    tensor_id_to_arg_idx: Dict[int, int]
    graph_input_tensor_ids: List[int]
    graph_input_ivalues: List[Any]
    def __call__(self, args): ...
    def __init__(self, tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_ivalues) -> None: ...

class ReturnValueHandler:
    index: Incomplete
    total_count: Incomplete
    def __init__(self, lazy_out_list) -> None: ...
    def duplicate_eager_tensors(self, eager_tensor_list): ...

def force_lazy_device(model: fx.GraphModule): ...
def get_fallback_ops(): ...
def extract_compiled_graph(model: fx.GraphModule, example_inputs) -> Callable: ...