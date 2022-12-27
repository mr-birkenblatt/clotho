# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.nn.parameter import Parameter as Parameter

from .. import init as init
from .module import Module as Module


class Embedding(Module):
    __constants__: Incomplete
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        padding_idx: Optional[int] = ..., max_norm: Optional[float] = ...,
        norm_type: float = ..., scale_grad_by_freq: bool = ...,
        sparse: bool = ..., _weight: Optional[Tensor] = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...

    @classmethod
    def from_pretrained(
        cls, embeddings, freeze: bool = ...,
        padding_idx: Incomplete | None = ...,
        max_norm: Incomplete | None = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., sparse: bool = ...): ...


class EmbeddingBag(Module):
    __constants__: Incomplete
    num_embeddings: int
    embedding_dim: int
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: Optional[int]

    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        max_norm: Optional[float] = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ...,
        _weight: Optional[Tensor] = ..., include_last_offset: bool = ...,
        padding_idx: Optional[int] = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def reset_parameters(self) -> None: ...

    def forward(
        self, input: Tensor, offsets: Optional[Tensor] = ...,
        per_sample_weights: Optional[Tensor] = ...) -> Tensor: ...

    def extra_repr(self) -> str: ...

    @classmethod
    def from_pretrained(
        cls, embeddings: Tensor, freeze: bool = ...,
        max_norm: Optional[float] = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ...,
        include_last_offset: bool = ...,
        padding_idx: Optional[int] = ...) -> EmbeddingBag: ...
