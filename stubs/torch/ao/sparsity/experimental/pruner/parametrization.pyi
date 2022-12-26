from _typeshed import Incomplete
from torch import nn as nn


class PruningParametrization(nn.Module):
    original_outputs: Incomplete
    pruned_outputs: Incomplete
    def __init__(self, original_outputs) -> None: ...
    def forward(self, x): ...

class ZeroesParametrization(nn.Module):
    original_outputs: Incomplete
    pruned_outputs: Incomplete
    def __init__(self, original_outputs) -> None: ...
    def forward(self, x): ...

class ActivationReconstruction:
    param: Incomplete
    def __init__(self, parametrization) -> None: ...
    def __call__(self, module, input, output): ...

class BiasHook:
    param: Incomplete
    prune_bias: Incomplete
    def __init__(self, parametrization, prune_bias) -> None: ...
    def __call__(self, module, input, output): ...
