import torch
from _typeshed import Incomplete


class MkldnnLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype) -> None: ...
    def forward(self, x): ...


class _MkldnnConvNd(torch.jit.ScriptModule):
    __constants__: Incomplete
    stride: Incomplete
    padding: Incomplete
    dilation: Incomplete
    groups: Incomplete
    def __init__(self, dense_module) -> None: ...
    def forward(self, x): ...


class MkldnnConv1d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None: ...


class MkldnnConv2d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None: ...


class MkldnnConv3d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype) -> None: ...


class MkldnnBatchNorm(torch.jit.ScriptModule):
    __constants__: Incomplete
    exponential_average_factor: float
    eps: Incomplete
    def __init__(self, dense_module) -> None: ...
    def forward(self, x): ...


class MkldnnPrelu(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype) -> None: ...
    def forward(self, x): ...


def to_mkldnn(module, dtype=...): ...
