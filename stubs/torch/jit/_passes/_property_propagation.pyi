from typing import Any, List

from torch import TensorType as TensorType
from torch._C import Graph as Graph


def apply_input_props_using_example(graph: Graph, example_input: List[Any]): ...
