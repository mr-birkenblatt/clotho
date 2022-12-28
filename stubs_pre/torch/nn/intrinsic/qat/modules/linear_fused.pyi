# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch.nn as nn
import torch.nn.intrinsic as nni
from _typeshed import Incomplete
from torch.nn import init as init
from torch.nn.parameter import Parameter as Parameter
from torch.nn.utils.fusion import (
    fuse_linear_bn_weights as fuse_linear_bn_weights,
)


class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    qconfig: Incomplete
    freeze_bn: Incomplete
    bn: Incomplete
    weight_fake_quant: Incomplete
    bias: Incomplete

    def __init__(
        self, in_features, out_features, bias: bool = ..., eps: float = ...,
        momentum: float = ..., freeze_bn: bool = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def reset_running_stats(self) -> None: ...
    def reset_bn_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def update_bn_stats(self): ...
    def freeze_bn_stats(self): ...
    def forward(self, input): ...
    training: Incomplete
    def train(self, mode: bool = ...): ...
    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...
