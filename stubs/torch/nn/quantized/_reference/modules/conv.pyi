from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.nn.common_types import _size_1_t

from .utils import ReferenceQuantizedModule as ReferenceQuantizedModule


class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    __annotations__: Incomplete
    @staticmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv1d(_ConvNd, nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = ..., padding: _size_1_t = ..., dilation: _size_1_t = ..., groups: int = ..., bias: bool = ..., padding_mode: str = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv2d(_ConvNd, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = ..., padding: int = ..., dilation: int = ..., groups: int = ..., bias: bool = ..., padding_mode: str = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class Conv3d(_ConvNd, nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = ..., padding: int = ..., dilation: int = ..., groups: int = ..., bias: bool = ..., padding_mode: str = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    @staticmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = ..., padding: _size_1_t = ..., output_padding: _size_1_t = ..., groups: int = ..., bias: bool = ..., dilation: _size_1_t = ..., padding_mode: str = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = ...) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = ..., padding: int = ..., output_padding: int = ..., groups: int = ..., bias: bool = ..., dilation: int = ..., padding_mode: str = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = ...) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...

class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = ..., padding: int = ..., output_padding: int = ..., groups: int = ..., bias: bool = ..., dilation: int = ..., padding_mode: str = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...
    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = ...) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, float_conv, weight_qparams): ...