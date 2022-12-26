from typing import Optional

from torch.types import _TensorOrTensors

from .function import Function as Function
from .variable import Variable as Variable


def backward(tensors: _TensorOrTensors, grad_tensors: Optional[_TensorOrTensors] = ..., retain_graph: Optional[bool] = ..., create_graph: bool = ..., grad_variables: Optional[_TensorOrTensors] = ..., inputs: Optional[_TensorOrTensors] = ...) -> None: ...

# Names in __all__ with no definition:
#   grad_mode
