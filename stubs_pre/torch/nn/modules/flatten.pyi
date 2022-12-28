# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Union

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.types import _size

from .module import Module as Module


class Flatten(Module):
    __constants__: Incomplete
    start_dim: int
    end_dim: int
    def __init__(self, start_dim: int = ..., end_dim: int = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Unflatten(Module):
    NamedShape: Incomplete
    __constants__: Incomplete
    dim: Union[int, str]
    unflattened_size: Union[_size, NamedShape]

    def __init__(
        self, dim: Union[int, str], unflattened_size: Union[_size,
                NamedShape]) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
