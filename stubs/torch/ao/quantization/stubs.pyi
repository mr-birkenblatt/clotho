# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


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
