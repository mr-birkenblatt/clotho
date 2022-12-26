from enum import Enum

import torch
from _typeshed import Incomplete

from .common import (
    amp_definitely_not_available as amp_definitely_not_available,
)


class _MultiDeviceReplicator:
    master: Incomplete
    def __init__(self, master_tensor: torch.Tensor) -> None: ...
    def get(self, device) -> torch.Tensor: ...

class OptState(Enum):
    READY: int
    UNSCALED: int
    STEPPED: int

class GradScaler:
    def __init__(self, init_scale=..., growth_factor: float = ..., backoff_factor: float = ..., growth_interval: int = ..., enabled: bool = ...) -> None: ...
    def scale(self, outputs): ...
    def unscale_(self, optimizer) -> None: ...
    def step(self, optimizer, *args, **kwargs): ...
    def update(self, new_scale: Incomplete | None = ...) -> None: ...
    def get_scale(self): ...
    def get_growth_factor(self): ...
    def set_growth_factor(self, new_factor) -> None: ...
    def get_backoff_factor(self): ...
    def set_backoff_factor(self, new_factor) -> None: ...
    def get_growth_interval(self): ...
    def set_growth_interval(self, new_interval) -> None: ...
    def is_enabled(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...