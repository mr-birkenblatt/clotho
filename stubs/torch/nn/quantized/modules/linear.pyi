# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional

import torch
from _typeshed import Incomplete
from torch.nn.quantized.modules.utils import (
    hide_packed_params_repr as hide_packed_params_repr,
)
from torch.nn.quantized.modules.utils import (
    WeightedQuantizedModule as WeightedQuantizedModule,
)
from torch.nn.utils.fusion import (
    fuse_linear_bn_weights as fuse_linear_bn_weights,
)
from torch.nn.utils.parametrize import (
    type_before_parametrizations as type_before_parametrizations,
)


class LinearPackedParams(torch.nn.Module):
    dtype: Incomplete
    def __init__(self, dtype=...) -> None: ...

    def set_weight_bias(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None: ...

    def forward(self, x): ...


class Linear(WeightedQuantizedModule):
    in_features: Incomplete
    out_features: Incomplete
    scale: float
    zero_point: int

    def __init__(
        self, in_features, out_features, bias_: bool = ...,
        dtype=...) -> None: ...

    def extra_repr(self): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def weight(self): ...
    def bias(self): ...

    def set_weight_bias(
        self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None: ...

    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_qlinear, output_scale, output_zero_point): ...
