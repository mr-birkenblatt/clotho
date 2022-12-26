from _typeshed import Incomplete
from torch import nn as nn


class QuantStub(nn.Module):
    qconfig: Incomplete
    def __init__(self, qconfig: Incomplete | None = ...) -> None: ...
    def forward(self, x): ...

class DeQuantStub(nn.Module):
    qconfig: Incomplete
    def __init__(self, qconfig: Incomplete | None = ...) -> None: ...
    def forward(self, x): ...

class QuantWrapper(nn.Module):
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module
    def __init__(self, module) -> None: ...
    def forward(self, X): ...
