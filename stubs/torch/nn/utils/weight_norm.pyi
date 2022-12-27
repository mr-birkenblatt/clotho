# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, TypeVar

from torch import norm_except_dim as norm_except_dim
from torch.nn.parameter import Parameter as Parameter
from torch.nn.parameter import UninitializedParameter as UninitializedParameter

from ..modules import Module as Module


class WeightNorm:
    name: str
    dim: int
    def __init__(self, name: str, dim: int) -> None: ...
    def compute_weight(self, module: Module) -> Any: ...
    @staticmethod
    def apply(module, name: str, dim: int) -> WeightNorm: ...
    def remove(self, module: Module) -> None: ...
    def __call__(self, module: Module, inputs: Any) -> None: ...


T_module = TypeVar('T_module', bound=Module)


def weight_norm(
    module: T_module, name: str = ..., dim: int = ...) -> T_module: ...


def remove_weight_norm(module: T_module, name: str = ...) -> T_module: ...
