import torch.nn as nn
from _typeshed import Incomplete
from torch.nn.intrinsic import LinearReLU as LinearReLU
from torch.nn.utils.parametrize import is_parametrized as is_parametrized
from torch.nn.utils.parametrize import (
    transfer_parametrizations_and_params as transfer_parametrizations_and_params,
)
from torch.nn.utils.parametrize import (
    type_before_parametrizations as type_before_parametrizations,
)


class Linear(nn.Linear):
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, in_features, out_features, bias: bool = ..., qconfig: Incomplete | None = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ...) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...
