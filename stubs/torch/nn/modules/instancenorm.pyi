from _typeshed import Incomplete
from torch import Tensor as Tensor

from .batchnorm import _LazyNormBase, _NormBase


class _InstanceNorm(_NormBase):
    def __init__(self, num_features: int, eps: float = ..., momentum: float = ..., affine: bool = ..., track_running_stats: bool = ..., device: Incomplete | None = ..., dtype: Incomplete | None = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class InstanceNorm1d(_InstanceNorm): ...

class LazyInstanceNorm1d(_LazyNormBase, _InstanceNorm):
    cls_to_become: Incomplete

class InstanceNorm2d(_InstanceNorm): ...

class LazyInstanceNorm2d(_LazyNormBase, _InstanceNorm):
    cls_to_become: Incomplete

class InstanceNorm3d(_InstanceNorm): ...

class LazyInstanceNorm3d(_LazyNormBase, _InstanceNorm):
    cls_to_become: Incomplete