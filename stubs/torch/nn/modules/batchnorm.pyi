# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.nn.parameter import Parameter as Parameter
from torch.nn.parameter import UninitializedBuffer as UninitializedBuffer
from torch.nn.parameter import UninitializedParameter as UninitializedParameter

from .. import init as init
from .lazy import LazyModuleMixin as LazyModuleMixin
from .module import Module as Module


class _NormBase(Module):
    __constants__: Incomplete
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    weight: Incomplete
    bias: Incomplete
    running_mean: Incomplete
    running_var: Incomplete
    num_batches_tracked: Incomplete

    def __init__(
        self, num_features: int, eps: float = ..., momentum: float = ...,
        affine: bool = ..., track_running_stats: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_running_stats(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self): ...


class _BatchNorm(_NormBase):

    def __init__(
        self, num_features: int, eps: float = ..., momentum: float = ...,
        affine: bool = ..., track_running_stats: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...


class _LazyNormBase(LazyModuleMixin, _NormBase):
    weight: UninitializedParameter
    bias: UninitializedParameter
    affine: Incomplete
    track_running_stats: Incomplete
    running_mean: Incomplete
    running_var: Incomplete
    num_batches_tracked: Incomplete

    def __init__(
        self, eps: float = ..., momentum: float = ..., affine: bool = ...,
        track_running_stats: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    num_features: Incomplete
    def initialize_parameters(self, input) -> None: ...


class BatchNorm1d(_BatchNorm):
    ...


class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    cls_to_become: Incomplete


class BatchNorm2d(_BatchNorm):
    ...


class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    cls_to_become: Incomplete


class BatchNorm3d(_BatchNorm):
    ...


class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    cls_to_become: Incomplete


class SyncBatchNorm(_BatchNorm):
    process_group: Incomplete

    def __init__(
        self, num_features: int, eps: float = ..., momentum: float = ...,
        affine: bool = ..., track_running_stats: bool = ...,
        process_group: Optional[Any] = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...

    @classmethod
    def convert_sync_batchnorm(
        cls, module, process_group: Incomplete | None = ...): ...
