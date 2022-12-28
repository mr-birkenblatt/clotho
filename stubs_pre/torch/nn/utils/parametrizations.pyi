# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from enum import Enum
from typing import Optional

import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..modules import Module as Module
from ..utils import parametrize as parametrize


class _OrthMaps(Enum):
    matrix_exp: Incomplete
    cayley: Incomplete
    householder: Incomplete


class _Orthogonal(Module):
    base: Tensor
    shape: Incomplete
    orthogonal_map: Incomplete

    def __init__(
        self, weight, orthogonal_map: _OrthMaps, *,
        use_trivialization: bool = ...) -> None: ...

    def forward(self, X: torch.Tensor) -> torch.Tensor: ...
    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor: ...


def orthogonal(
    module: Module, name: str = ..., orthogonal_map: Optional[str] = ..., *,
    use_trivialization: bool = ...) -> Module: ...


class _SpectralNorm(Module):
    dim: Incomplete
    eps: Incomplete
    n_power_iterations: Incomplete

    def __init__(
        self, weight: torch.Tensor, n_power_iterations: int = ...,
        dim: int = ..., eps: float = ...) -> None: ...

    def forward(self, weight: torch.Tensor) -> torch.Tensor: ...
    def right_inverse(self, value: torch.Tensor) -> torch.Tensor: ...


def spectral_norm(
    module: Module, name: str = ..., n_power_iterations: int = ...,
    eps: float = ..., dim: Optional[int] = ...) -> Module: ...
