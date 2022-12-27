# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Dict, Optional

import torch.nn as nn
from _typeshed import Incomplete
from torch import Tensor as Tensor

from .utils import ReferenceQuantizedModule as ReferenceQuantizedModule


class Embedding(nn.Embedding, ReferenceQuantizedModule):

    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        padding_idx: Optional[int] = ..., max_norm: Optional[float] = ...,
        norm_type: float = ..., scale_grad_by_freq: bool = ...,
        sparse: bool = ..., _weight: Optional[Tensor] = ...,
        device: Incomplete | None = ..., dtype: Incomplete | None = ...,
        weight_qparams: Optional[Dict[str, Any]] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams): ...


class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):

    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        max_norm: Optional[float] = ..., norm_type: float = ...,
        scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ...,
        _weight: Optional[Tensor] = ..., include_last_offset: bool = ...,
        padding_idx: Optional[int] = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ..., weight_qparams: Optional[Dict[str,
                Any]] = ...) -> None: ...

    def forward(
        self, input: Tensor, offsets: Optional[Tensor] = ...,
        per_sample_weights: Optional[Tensor] = ...) -> Tensor: ...

    @classmethod
    def from_float(cls, mod, weight_qparams): ...
