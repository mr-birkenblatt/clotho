# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Dict, Optional, Tuple

from _typeshed import Incomplete
from torch import nn
from torch import Tensor as Tensor
from torch.nn.utils.rnn import PackedSequence as PackedSequence


def apply_permutation(
    tensor: Tensor, permutation: Tensor, dim: int = ...) -> Tensor: ...


def get_weight_and_quantization_params(module, wn): ...


def get_quantized_weight(module, wn): ...


def get_quantize_and_dequantized_weight(module, wn): ...


class RNNCellBase(nn.RNNCellBase):

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks: int,
        device: Incomplete | None = ..., dtype: Incomplete | None = ...,
        weight_qparams_dict: Incomplete | None = ...) -> None: ...

    def get_quantized_weight_ih(self): ...
    def get_quantized_weight_hh(self): ...
    def get_weight_ih(self): ...
    def get_weight_hh(self): ...


class RNNCell(RNNCellBase):
    nonlinearity: Incomplete

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ...,
        nonlinearity: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...,
        weight_qparams_dict: Optional[Dict[str, Dict[
                                str, Any]]] = ...) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict): ...


class LSTMCell(RNNCellBase):

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ...,
        device: Incomplete | None = ..., dtype: Incomplete | None = ...,
        weight_qparams_dict: Optional[Dict[str, Dict[
                                str, Any]]] = ...) -> None: ...

    def forward(
        self, input: Tensor, hx: Optional[Tuple[
                        Tensor, Tensor]] = ...) -> Tuple[Tensor, Tensor]: ...

    @classmethod
    def from_float(cls, mod, weight_qparams_dict): ...


class GRUCell(RNNCellBase):

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ...,
        device: Incomplete | None = ..., dtype: Incomplete | None = ...,
        weight_qparams_dict: Optional[Dict[str, Dict[
                                str, Any]]] = ...) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict): ...


class RNNBase(nn.RNNBase):

    def __init__(
        self, mode: str, input_size: int, hidden_size: int,
        num_layers: int = ..., bias: bool = ..., batch_first: bool = ...,
        dropout: float = ..., bidirectional: bool = ...,
        proj_size: int = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...,
        weight_qparams_dict: Optional[Dict[str, Dict[
                                str, Any]]] = ...) -> None: ...


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None: ...

    def permute_hidden(
        self, hx: Tuple[Tensor, Tensor], permutation: Optional[
                Tensor]) -> Tuple[Tensor, Tensor]: ...

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[
            int, int, int]: ...

    def check_forward_args(
        self, input: Tensor, hidden: Tuple[Tensor, Tensor],
        batch_sizes: Optional[Tensor]): ...

    def get_quantized_weight_bias_dict(self): ...
    def get_flat_weights(self): ...
    def forward(self, input, hx: Incomplete | None = ...): ...
    @classmethod
    def from_float(cls, mod, weight_qparams_dict): ...
