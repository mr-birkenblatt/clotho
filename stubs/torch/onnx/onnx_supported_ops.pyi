from typing import Union

from _typeshed import Incomplete
from torch import _C
from torch.onnx import symbolic_registry as symbolic_registry


class _TorchSchema:
    name: Incomplete
    overload_name: Incomplete
    arguments: Incomplete
    optional_arguments: Incomplete
    returns: Incomplete
    opsets: Incomplete
    def __init__(self, schema: Union[_C.FunctionSchema, str]) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other) -> bool: ...
    def is_aten(self) -> bool: ...
    def is_backward(self) -> bool: ...

def onnx_supported_ops(): ...
