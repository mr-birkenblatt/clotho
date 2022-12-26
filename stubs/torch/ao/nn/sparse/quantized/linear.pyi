from typing import Optional

import torch
from _typeshed import Incomplete
from torch.nn.quantized.modules.utils import (
    hide_packed_params_repr as hide_packed_params_repr,
)


class LinearPackedParams(torch.nn.Module):
    prepack_op: Incomplete
    unpack_op: Incomplete
    dtype: Incomplete
    def __init__(self, row_block_size: int = ..., col_block_size: int = ..., dtype=...) -> None: ...
    weight: Incomplete
    bias: Incomplete
    row_block_size: Incomplete
    col_block_size: Incomplete
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor], row_block_size: Optional[int], col_block_size: Optional[int]) -> None: ...
    def forward(self, x): ...

class Linear(torch.nn.Module):
    in_features: Incomplete
    out_features: Incomplete
    scale: float
    zero_point: int
    def __init__(self, in_features, out_features, row_block_size, col_block_size, bias: bool = ..., dtype=...) -> None: ...
    def extra_repr(self): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def weight(self): ...
    def bias(self): ...
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor], row_block_size: Optional[int], col_block_size: Optional[int]) -> None: ...
    @classmethod
    def from_float(cls, mod): ...
