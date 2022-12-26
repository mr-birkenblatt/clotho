from typing import Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..common_types import _ratio_2_t, _ratio_any_t, _size_2_t, _size_any_t
from .module import Module as Module


class Upsample(Module):
    __constants__: Incomplete
    name: str
    size: Optional[_size_any_t]
    scale_factor: Optional[_ratio_any_t]
    mode: str
    align_corners: Optional[bool]
    recompute_scale_factor: Optional[bool]
    def __init__(self, size: Optional[_size_any_t] = ..., scale_factor: Optional[_ratio_any_t] = ..., mode: str = ..., align_corners: Optional[bool] = ..., recompute_scale_factor: Optional[bool] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

class UpsamplingNearest2d(Upsample):
    def __init__(self, size: Optional[_size_2_t] = ..., scale_factor: Optional[_ratio_2_t] = ...) -> None: ...

class UpsamplingBilinear2d(Upsample):
    def __init__(self, size: Optional[_size_2_t] = ..., scale_factor: Optional[_ratio_2_t] = ...) -> None: ...
