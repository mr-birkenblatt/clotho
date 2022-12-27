# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, List, Optional, Tuple, Union

import torch
from _typeshed import Incomplete

from ._lowrank import pca_lowrank as pca_lowrank
from ._lowrank import svd_lowrank as svd_lowrank


def broadcast_tensors(*tensors): ...


def broadcast_shapes(*shapes): ...


def split(
    tensor: Tensor, split_size_or_sections: Union[int, List[int]],
    dim: int = ...) -> List[Tensor]: ...


def einsum(*args: Any) -> Tensor: ...


def meshgrid(
    *tensors: Union[Tensor, List[Tensor]], indexing: Optional[
            str] = ...) -> Tuple[Tensor, ...]: ...


def stft(
    input: Tensor, n_fft: int, hop_length: Optional[int] = ...,
    win_length: Optional[int] = ..., window: Optional[Tensor] = ...,
    center: bool = ..., pad_mode: str = ..., normalized: bool = ...,
    onesided: Optional[bool] = ..., return_complex: Optional[
            bool] = ...) -> Tensor: ...


istft: Incomplete
unique: Incomplete
unique_consecutive: Incomplete


def tensordot(a, b, dims: int = ..., out: Optional[torch.Tensor] = ...): ...


def cartesian_prod(*tensors): ...


def block_diag(*tensors): ...


def cdist(
    x1: Tensor, x2: Tensor, p: float = ...,
    compute_mode: str = ...) -> Tensor: ...


def atleast_1d(*tensors): ...


def atleast_2d(*tensors): ...


def atleast_3d(*tensors): ...


def norm(
    input, p: str = ..., dim: Incomplete | None = ..., keepdim: bool = ...,
    out: Incomplete | None = ..., dtype: Incomplete | None = ...): ...


def chain_matmul(*matrices, out: Incomplete | None = ...): ...


lu: Incomplete


def align_tensors(*tensors) -> None: ...
