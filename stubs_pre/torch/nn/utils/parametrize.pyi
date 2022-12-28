# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from collections.abc import Generator
from typing import Optional, Sequence, Union

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.nn.modules.container import Module as Module
from torch.nn.modules.container import ModuleDict as ModuleDict
from torch.nn.modules.container import ModuleList as ModuleList
from torch.nn.parameter import Parameter as Parameter


def cached() -> Generator[None, None, None]: ...


class ParametrizationList(ModuleList):
    original: Tensor
    unsafe: bool
    is_tensor: Incomplete
    ntensors: Incomplete

    def __init__(
        self, modules: Sequence[Module], original: Union[Tensor, Parameter],
        unsafe: bool = ...) -> None: ...

    def right_inverse(self, value: Tensor) -> None: ...
    def forward(self) -> Tensor: ...


def register_parametrization(
    module: Module, tensor_name: str, parametrization: Module, *,
    unsafe: bool = ...) -> Module: ...


def is_parametrized(
    module: Module, tensor_name: Optional[str] = ...) -> bool: ...


def remove_parametrizations(
    module: Module, tensor_name: str,
    leave_parametrized: bool = ...) -> Module: ...


def type_before_parametrizations(module: Module) -> type: ...


def transfer_parametrizations_and_params(
    from_module: Module, to_module: Module,
    tensor_name: Optional[str] = ...) -> Module: ...
