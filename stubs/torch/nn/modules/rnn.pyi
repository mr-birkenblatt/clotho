# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import List, Optional, overload, Tuple

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .. import init as init
from ..parameter import Parameter as Parameter
from ..utils.rnn import PackedSequence as PackedSequence
from .module import Module as Module


def apply_permutation(
    tensor: Tensor, permutation: Tensor, dim: int = ...) -> Tensor: ...


class RNNBase(Module):
    __constants__: Incomplete
    __jit_unused_properties__: Incomplete
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    def __init__(
        self, mode: str, input_size: int, hidden_size: int,
        num_layers: int = ..., bias: bool = ..., batch_first: bool = ...,
        dropout: float = ..., bidirectional: bool = ...,
        proj_size: int = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...): ...

    def __setattr__(self, attr, value) -> None: ...
    def flatten_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...

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
        batch_sizes: Optional[Tensor]): ...

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]): ...
    def extra_repr(self) -> str: ...
    @property
    def all_weights(self) -> List[List[Parameter]]: ...


class RNN(RNNBase):
    nonlinearity: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

    @overload
    def forward(
        self, input: Tensor, hx: Optional[Tensor] = ...) -> Tuple[
            Tensor, Tensor]: ...

    @overload
    def forward(
        self, input: PackedSequence, hx: Optional[Tensor] = ...) -> Tuple[
            PackedSequence, Tensor]: ...


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None: ...

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[
            int, int, int]: ...

    def check_forward_args(
        self, input: Tensor, hidden: Tuple[Tensor, Tensor],
        batch_sizes: Optional[Tensor]): ...

    def permute_hidden(
        self, hx: Tuple[Tensor, Tensor], permutation: Optional[
                Tensor]) -> Tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self, input: Tensor, hx: Optional[Tuple[
                        Tensor, Tensor]] = ...) -> Tuple[Tensor, Tuple[
                    Tensor, Tensor]]: ...

    @overload
    def forward(
        self, input: PackedSequence, hx: Optional[Tuple[Tensor,
                        Tensor]] = ...) -> Tuple[PackedSequence, Tuple[
                    Tensor, Tensor]]: ...


class GRU(RNNBase):
    def __init__(self, *args, **kwargs) -> None: ...

    @overload
    def forward(
        self, input: Tensor, hx: Optional[Tensor] = ...) -> Tuple[
            Tensor, Tensor]: ...

    @overload
    def forward(
        self, input: PackedSequence, hx: Optional[Tensor] = ...) -> Tuple[
            PackedSequence, Tensor]: ...


class RNNCellBase(Module):
    __constants__: Incomplete
    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Incomplete
    bias_hh: Incomplete

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks: int,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def extra_repr(self) -> str: ...
    def reset_parameters(self) -> None: ...


class RNNCell(RNNCellBase):
    __constants__: Incomplete
    nonlinearity: str

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ...,
        nonlinearity: str = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...


class LSTMCell(RNNCellBase):

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, input: Tensor, hx: Optional[Tuple[
                        Tensor, Tensor]] = ...) -> Tuple[Tensor, Tensor]: ...


class GRUCell(RNNCellBase):

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...
