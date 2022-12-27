# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


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
