from typing import Any, Optional, TypeVar

import torch
from _typeshed import Incomplete
from torch.nn.functional import normalize as normalize

from ..modules import Module as Module


class SpectralNorm:
    name: str
    dim: int
    n_power_iterations: int
    eps: float
    def __init__(self, name: str = ..., n_power_iterations: int = ..., dim: int = ..., eps: float = ...) -> None: ...
    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor: ...
    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor: ...
    def remove(self, module: Module) -> None: ...
    def __call__(self, module: Module, inputs: Any) -> None: ...
    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> SpectralNorm: ...

class SpectralNormLoadStateDictPreHook:
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...

class SpectralNormStateDictHook:
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self, module, state_dict, prefix, local_metadata) -> None: ...
T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module, name: str = ..., n_power_iterations: int = ..., eps: float = ..., dim: Optional[int] = ...) -> T_module: ...
def remove_spectral_norm(module: T_module, name: str = ...) -> T_module: ...