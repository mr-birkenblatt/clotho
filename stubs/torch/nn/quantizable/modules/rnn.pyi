# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional, Tuple

import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor


class LSTMCell(torch.nn.Module):
    input_size: Incomplete
    hidden_size: Incomplete
    bias: Incomplete
    igates: Incomplete
    hgates: Incomplete
    gates: Incomplete
    fgate_cx: Incomplete
    igate_cgate: Incomplete
    fgate_cx_igate_cgate: Incomplete
    ogate_cy: Incomplete

    def __init__(
        self, input_dim: int, hidden_dim: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor,
        Tensor]] = ...) -> Tuple[Tensor, Tensor]: ...

    def initialize_hidden(
        self, batch_size: int, is_quantized: bool = ...) -> Tuple[Tensor,
            Tensor]: ...

    @classmethod
    def from_params(
        cls, wi, wh, bi: Incomplete | None = ...,
        bh: Incomplete | None = ...): ...

    @classmethod
    def from_float(cls, other): ...


class _LSTMSingleLayer(torch.nn.Module):
    cell: Incomplete

    def __init__(
        self, input_dim: int, hidden_dim: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = ...): ...

    @classmethod
    def from_params(cls, *args, **kwargs): ...


class _LSTMLayer(torch.nn.Module):
    batch_first: Incomplete
    bidirectional: Incomplete
    layer_fw: Incomplete
    layer_bw: Incomplete

    def __init__(
        self, input_dim: int, hidden_dim: int, bias: bool = ...,
        batch_first: bool = ..., bidirectional: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = ...): ...

    @classmethod
    def from_float(
        cls, other, layer_idx: int = ..., qconfig: Incomplete | None = ...,
        **kwargs): ...


class LSTM(torch.nn.Module):
    input_size: Incomplete
    hidden_size: Incomplete
    num_layers: Incomplete
    bias: Incomplete
    batch_first: Incomplete
    dropout: Incomplete
    bidirectional: Incomplete
    training: bool
    layers: Incomplete

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = ...,
        bias: bool = ..., batch_first: bool = ..., dropout: float = ...,
        bidirectional: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = ...): ...

    @classmethod
    def from_float(cls, other, qconfig: Incomplete | None = ...): ...
    @classmethod
    def from_observed(cls, other): ...
