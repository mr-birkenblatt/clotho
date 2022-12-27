# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Optional, Union

from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..init import xavier_uniform_ as xavier_uniform_
from .activation import MultiheadAttention as MultiheadAttention
from .container import ModuleList as ModuleList
from .dropout import Dropout as Dropout
from .linear import Linear as Linear
from .module import Module as Module
from .normalization import LayerNorm as LayerNorm


class Transformer(Module):
    encoder: Incomplete
    decoder: Incomplete
    d_model: Incomplete
    nhead: Incomplete
    batch_first: Incomplete

    def __init__(
        self, d_model: int = ..., nhead: int = ...,
        num_encoder_layers: int = ..., num_decoder_layers: int = ...,
        dim_feedforward: int = ..., dropout: float = ...,
        activation: Union[str, Callable[[Tensor], Tensor]] = ...,
        custom_encoder: Optional[Any] = ...,
        custom_decoder: Optional[Any] = ..., layer_norm_eps: float = ...,
        batch_first: bool = ..., norm_first: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = ...,
        tgt_mask: Optional[Tensor] = ...,
        memory_mask: Optional[Tensor] = ...,
        src_key_padding_mask: Optional[Tensor] = ...,
        tgt_key_padding_mask: Optional[Tensor] = ...,
        memory_key_padding_mask: Optional[Tensor] = ...) -> Tensor: ...

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor: ...


class TransformerEncoder(Module):
    __constants__: Incomplete
    layers: Incomplete
    num_layers: Incomplete
    norm: Incomplete
    enable_nested_tensor: Incomplete

    def __init__(
        self, encoder_layer, num_layers, norm: Incomplete | None = ...,
        enable_nested_tensor: bool = ...) -> None: ...

    def forward(
        self, src: Tensor, mask: Optional[Tensor] = ...,
        src_key_padding_mask: Optional[Tensor] = ...) -> Tensor: ...


class TransformerDecoder(Module):
    __constants__: Incomplete
    layers: Incomplete
    num_layers: Incomplete
    norm: Incomplete

    def __init__(
        self, decoder_layer, num_layers,
        norm: Incomplete | None = ...) -> None: ...

    def forward(
        self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = ...,
        memory_mask: Optional[Tensor] = ...,
        tgt_key_padding_mask: Optional[Tensor] = ...,
        memory_key_padding_mask: Optional[Tensor] = ...) -> Tensor: ...


class TransformerEncoderLayer(Module):
    __constants__: Incomplete
    self_attn: Incomplete
    linear1: Incomplete
    dropout: Incomplete
    linear2: Incomplete
    norm_first: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    dropout1: Incomplete
    dropout2: Incomplete
    activation_relu_or_gelu: int
    activation: Incomplete

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = ...,
        dropout: float = ..., activation: Union[str, Callable[[Tensor],
                        Tensor]] = ..., layer_norm_eps: float = ...,
        batch_first: bool = ..., norm_first: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = ...,
        src_key_padding_mask: Optional[Tensor] = ...) -> Tensor: ...


class TransformerDecoderLayer(Module):
    __constants__: Incomplete
    self_attn: Incomplete
    multihead_attn: Incomplete
    linear1: Incomplete
    dropout: Incomplete
    linear2: Incomplete
    norm_first: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    norm3: Incomplete
    dropout1: Incomplete
    dropout2: Incomplete
    dropout3: Incomplete
    activation: Incomplete

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = ...,
        dropout: float = ..., activation: Union[str, Callable[[Tensor],
                        Tensor]] = ..., layer_norm_eps: float = ...,
        batch_first: bool = ..., norm_first: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = ...,
        memory_mask: Optional[Tensor] = ...,
        tgt_key_padding_mask: Optional[Tensor] = ...,
        memory_key_padding_mask: Optional[Tensor] = ...) -> Tensor: ...
