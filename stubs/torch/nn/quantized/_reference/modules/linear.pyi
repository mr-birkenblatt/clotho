from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .utils import ReferenceQuantizedModule as ReferenceQuantizedModule


class Linear(nn.Linear, ReferenceQuantizedModule):
    def __init__(self, in_features: int, out_features: int, bias_: bool = ..., device: Optional[torch.device] = ..., dtype: Optional[torch.dtype] = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_linear, weight_qparams): ...
