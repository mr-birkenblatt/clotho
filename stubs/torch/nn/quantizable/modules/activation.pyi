# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional, Tuple

from _typeshed import Incomplete
from torch import nn as nn
from torch import Tensor as Tensor


class MultiheadAttention(nn.MultiheadAttention):
    __constants__: Incomplete
    linear_Q: Incomplete
    linear_K: Incomplete
    linear_V: Incomplete
    out_proj: Incomplete
    q_scaling_product: Incomplete
    quant_attn_output: Incomplete
    quant_attn_output_weights: Incomplete
    dequant_q: Incomplete
    dequant_k: Incomplete
    dequant_v: Incomplete

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = ...,
        bias: bool = ..., add_bias_kv: bool = ..., add_zero_attn: bool = ...,
        kdim: int = ..., vdim: int = ..., batch_first: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    @classmethod
    def from_float(cls, other): ...
    def dequantize(self): ...
    @classmethod
    def from_observed(cls, other): ...

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        key_padding_mask: Optional[Tensor] = ..., need_weights: bool = ...,
        attn_mask: Optional[Tensor] = ...,
        average_attn_weights: bool = ...) -> Tuple[Tensor, Optional[
                    Tensor]]: ...
