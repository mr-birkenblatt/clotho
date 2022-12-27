# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch.nn as nn
from _typeshed import Incomplete
from torch import Tensor as Tensor


class Embedding(nn.Embedding):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, num_embeddings, embedding_dim,
        padding_idx: Incomplete | None = ...,
        max_norm: Incomplete | None = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., sparse: bool = ...,
        _weight: Incomplete | None = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...,
        qconfig: Incomplete | None = ...) -> None: ...

    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...


class EmbeddingBag(nn.EmbeddingBag):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, num_embeddings, embedding_dim,
        max_norm: Incomplete | None = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ...,
        _weight: Incomplete | None = ..., include_last_offset: bool = ...,
        padding_idx: Incomplete | None = ...,
        qconfig: Incomplete | None = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, input, offsets: Incomplete | None = ...,
        per_sample_weights: Incomplete | None = ...) -> Tensor: ...

    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...
