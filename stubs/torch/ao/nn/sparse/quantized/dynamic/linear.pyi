from typing import Optional

import torch
from _typeshed import Incomplete
from torch.ao.nn.sparse.quantized import linear as linear
from torch.ao.nn.sparse.quantized.utils import (
    LinearBlockSparsePattern as LinearBlockSparsePattern,
)
from torch.nn.quantized.modules.utils import (
    hide_packed_params_repr as hide_packed_params_repr,
)


class Linear(torch.nn.Module):
    in_features: Incomplete
    out_features: Incomplete
    def __init__(self, in_features, out_features, row_block_size, col_block_size, bias: bool = ..., dtype=...) -> None: ...
    def extra_repr(self): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def weight(self): ...
    def bias(self): ...
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor], row_block_size: Optional[int], col_block_size: Optional[int]) -> None: ...
    @classmethod
    def from_float(cls, mod): ...
