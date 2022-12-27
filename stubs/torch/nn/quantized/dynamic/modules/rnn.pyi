# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._jit_internal import Dict as Dict
from torch._jit_internal import List as List
from torch._jit_internal import Optional as Optional
from torch._jit_internal import Tuple as Tuple
from torch._jit_internal import Union as Union
from torch.nn.utils.rnn import PackedSequence as PackedSequence


def apply_permutation(
    tensor: Tensor, permutation: Tensor, dim: int = ...) -> Tensor: ...


def pack_weight_bias(qweight, bias, dtype): ...


class PackedParameter(torch.nn.Module):
    param: Incomplete
    def __init__(self, param) -> None: ...


class RNNBase(torch.nn.Module):
    mode: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    num_layers: Incomplete
    bias: Incomplete
    batch_first: Incomplete
    dropout: Incomplete
    bidirectional: Incomplete
    dtype: Incomplete
    version: int
    training: bool

    def __init__(
        self, mode, input_size, hidden_size, num_layers: int = ...,
        bias: bool = ..., batch_first: bool = ..., dropout: float = ...,
        bidirectional: bool = ..., dtype=...) -> None: ...

    def extra_repr(self): ...

    def check_input(
        self, input: Tensor, batch_sizes: Optional[Tensor]) -> None: ...

    def get_expected_hidden_size(
        self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[
            int, int, int]: ...

    def check_hidden_size(
        self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
        msg: str = ...) -> None: ...

    def check_forward_args(
        self, input: Tensor, hidden: Tensor,
        batch_sizes: Optional[Tensor]) -> None: ...

    def permute_hidden(
        self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor: ...

    def set_weight_bias(self, weight_bias_dict): ...
    @classmethod
    def from_float(cls, mod): ...
    def get_weight(self): ...
    def get_bias(self): ...


class LSTM(RNNBase):
    __overloads__: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

    def forward_impl(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]],
        batch_sizes: Optional[Tensor], max_batch_size: int,
        sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tuple[
                    Tensor, Tensor]]: ...

    def forward_tensor(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = ...,
        ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]: ...

    def forward_packed(
        self, input: PackedSequence,
        hx: Optional[Tuple[Tensor, Tensor]] = ...,
        ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]: ...

    def permute_hidden(
        self, hx: Tuple[Tensor, Tensor],
        permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]: ...

    def check_forward_args(
        self, input: Tensor, hidden: Tuple[Tensor, Tensor],
        batch_sizes: Optional[Tensor]) -> None: ...

    def forward(self, input, hx: Incomplete | None = ...): ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_mod): ...


class GRU(RNNBase):
    __overloads__: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

    def check_forward_args(
        self, input: Tensor, hidden: Tensor,
        batch_sizes: Optional[Tensor]) -> None: ...

    def forward_impl(
        self, input: Tensor, hx: Optional[Tensor],
        batch_sizes: Optional[Tensor], max_batch_size: int,
        sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tensor]: ...

    def forward_tensor(
        self, input: Tensor, hx: Optional[Tensor] = ...) -> Tuple[
            Tensor, Tensor]: ...

    def forward_packed(
        self, input: PackedSequence, hx: Optional[Tensor] = ...) -> Tuple[
            PackedSequence, Tensor]: ...

    def permute_hidden(
        self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor: ...

    def forward(self, input, hx: Incomplete | None = ...): ...
    @classmethod
    def from_float(cls, mod): ...


class RNNCellBase(torch.nn.Module):
    __constants__: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    bias: Incomplete
    weight_dtype: Incomplete
    bias_ih: Incomplete
    bias_hh: Incomplete

    def __init__(
        self, input_size, hidden_size, bias: bool = ...,
        num_chunks: int = ..., dtype=...) -> None: ...

    def extra_repr(self): ...
    def check_forward_input(self, input) -> None: ...

    def check_forward_hidden(
        self, input: Tensor, hx: Tensor, hidden_label: str = ...) -> None: ...

    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_mod): ...
    def get_weight(self): ...
    def get_bias(self): ...
    def set_weight_bias(self, weight_bias_dict) -> None: ...


class RNNCell(RNNCellBase):
    __constants__: Incomplete
    nonlinearity: Incomplete

    def __init__(
        self, input_size, hidden_size, bias: bool = ...,
        nonlinearity: str = ..., dtype=...) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...
    @classmethod
    def from_float(cls, mod): ...


class LSTMCell(RNNCellBase):
    def __init__(self, *args, **kwargs) -> None: ...

    def forward(
        self, input: Tensor, hx: Optional[Tuple[
                        Tensor, Tensor]] = ...) -> Tuple[Tensor, Tensor]: ...

    @classmethod
    def from_float(cls, mod): ...


class GRUCell(RNNCellBase):

    def __init__(
        self, input_size, hidden_size, bias: bool = ...,
        dtype=...) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...
    @classmethod
    def from_float(cls, mod): ...
