# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch.nn.quantized.functional
from _typeshed import Incomplete


class LayerNorm(torch.nn.LayerNorm):
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, normalized_shape, weight, bias, scale, zero_point,
        eps: float = ..., elementwise_affine: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...


class GroupNorm(torch.nn.GroupNorm):
    __constants__: Incomplete
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, num_groups, num_channels, weight, bias, scale, zero_point,
        eps: float = ..., affine: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...


class InstanceNorm1d(torch.nn.InstanceNorm1d):
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, num_features, weight, bias, scale, zero_point,
        eps: float = ..., momentum: float = ..., affine: bool = ...,
        track_running_stats: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...


class InstanceNorm2d(torch.nn.InstanceNorm2d):
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, num_features, weight, bias, scale, zero_point,
        eps: float = ..., momentum: float = ..., affine: bool = ...,
        track_running_stats: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...


class InstanceNorm3d(torch.nn.InstanceNorm3d):
    weight: Incomplete
    bias: Incomplete

    def __init__(
        self, num_features, weight, bias, scale, zero_point,
        eps: float = ..., momentum: float = ..., affine: bool = ...,
        track_running_stats: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, mod, scale, zero_point): ...
