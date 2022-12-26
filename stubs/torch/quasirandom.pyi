from typing import Optional

import torch
from _typeshed import Incomplete


class SobolEngine:
    MAXBIT: int
    MAXDIM: int
    seed: Incomplete
    scramble: Incomplete
    dimension: Incomplete
    sobolstate: Incomplete
    shift: Incomplete
    quasi: Incomplete
    num_generated: int
    def __init__(self, dimension, scramble: bool = ..., seed: Incomplete | None = ...) -> None: ...
    def draw(self, n: int = ..., out: Optional[torch.Tensor] = ..., dtype: torch.dtype = ...) -> torch.Tensor: ...
    def draw_base2(self, m: int, out: Optional[torch.Tensor] = ..., dtype: torch.dtype = ...) -> torch.Tensor: ...
    def reset(self): ...
    def fast_forward(self, n): ...
