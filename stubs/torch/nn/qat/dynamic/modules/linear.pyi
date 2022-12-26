import torch
from _typeshed import Incomplete
from torch.ao.quantization import (
    activation_is_memoryless as activation_is_memoryless,
)


class Linear(torch.nn.qat.Linear):
    def __init__(self, in_features, out_features, bias: bool = ..., qconfig: Incomplete | None = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ...) -> None: ...
