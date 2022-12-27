# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._jit_internal import List as List
from torch._jit_internal import Optional as Optional
from torch.nn.quantized.modules.utils import (
    hide_packed_params_repr as hide_packed_params_repr,
)


class EmbeddingPackedParams(torch.nn.Module):
    dtype: Incomplete
    def __init__(self, num_embeddings, embedding_dim, dtype=...) -> None: ...
    def set_weight(self, weight: torch.Tensor) -> None: ...
    def forward(self, x): ...


class Embedding(torch.nn.Module):
    num_embeddings: Incomplete
    embedding_dim: Incomplete
    dtype: Incomplete

    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        padding_idx: Optional[int] = ..., max_norm: Optional[float] = ...,
        norm_type: float = ..., scale_grad_by_freq: bool = ...,
        sparse: bool = ..., _weight: Optional[Tensor] = ...,
        dtype=...) -> None: ...

    def forward(self, indices: Tensor) -> Tensor: ...
    def extra_repr(self): ...
    def set_weight(self, w: torch.Tensor) -> None: ...
    def weight(self): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_embedding): ...


class EmbeddingBag(Embedding):
    mode: Incomplete
    pruned_weights: bool
    include_last_offset: Incomplete
    dtype: Incomplete

    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        max_norm: Optional[float] = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ...,
        _weight: Optional[Tensor] = ..., include_last_offset: bool = ...,
        dtype=...) -> None: ...

    def forward(
        self, indices: Tensor, offsets: Optional[Tensor] = ...,
        per_sample_weights: Optional[Tensor] = ...,
        compressed_indices_mapping: Optional[Tensor] = ...) -> Tensor: ...

    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_embedding_bag): ...
