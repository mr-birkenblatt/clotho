import torch.nn.intrinsic as nni
import torch.nn.qat as nnqat
from _typeshed import Incomplete


class LinearReLU(nnqat.Linear, nni._FusedModule):
    def __init__(self, in_features, out_features, bias: bool = ..., qconfig: Incomplete | None = ...) -> None: ...
    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...