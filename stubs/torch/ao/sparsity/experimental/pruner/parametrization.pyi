# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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
