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
from torch.nn.utils.rnn import PackedSequence as PackedSequence


class QuantizedLinear(torch.jit.ScriptModule):
    __constants__: Incomplete
    in_features: Incomplete
    out_features: Incomplete
    weight: Incomplete
    col_offsets: Incomplete
    bias: Incomplete
    def __init__(self, other) -> None: ...
    def forward(self, input): ...
    def extra_repr(self): ...


class QuantizedLinearFP16(torch.jit.ScriptModule):
    in_features: Incomplete
    out_features: Incomplete
    original_weight: Incomplete
    weight: Incomplete
    bias: Incomplete
    def __init__(self, other) -> None: ...
    def forward(self, input): ...
    def extra_repr(self): ...


class QuantizedRNNCellBase(torch.jit.ScriptModule):
    __constants__: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    bias: Incomplete
    bias_ih: Incomplete
    bias_hh: Incomplete
    def __init__(self, other) -> None: ...
    def extra_repr(self): ...
    def check_forward_input(self, input) -> None: ...

    def check_forward_hidden(
        self, input: Tensor, hx: Tensor, hidden_label: str = ...) -> None: ...


class QuantizedRNNCell(QuantizedRNNCellBase):
    __constants__: Incomplete
    nonlinearity: Incomplete
    def __init__(self, other) -> None: ...
    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...


class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other) -> None: ...

    def forward(
        self, input: Tensor, hx: Optional[Tuple[
                        Tensor, Tensor]] = ...) -> Tuple[Tensor, Tensor]: ...


class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other) -> None: ...
    def forward(self, input: Tensor, hx: Optional[Tensor] = ...) -> Tensor: ...


def apply_permutation(
    tensor: Tensor, permutation: Tensor, dim: int = ...) -> Tensor: ...


class QuantizedRNNBase(torch.jit.ScriptModule):
    __constants__: Incomplete
    mode: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    num_layers: Incomplete
    bias: Incomplete
    batch_first: Incomplete
    dropout: Incomplete
    bidirectional: Incomplete
    dtype: Incomplete
    all_weights: Incomplete
    def __init__(self, other, dtype=...): ...

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


class QuantizedLSTM(QuantizedRNNBase):
    __overloads__: Incomplete
    def __init__(self, other, dtype) -> None: ...

    def forward_impl(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]],
        batch_sizes: Optional[Tensor], max_batch_size: int,
        sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tuple[
                    Tensor, Tensor]]: ...

    def forward_tensor(
        self, input: Tensor, hx: Optional[Tuple[
                        Tensor, Tensor]] = ...) -> Tuple[Tensor, Tuple[
                    Tensor, Tensor]]: ...

    def forward_packed(
        self, input: PackedSequence, hx: Optional[Tuple[Tensor,
                        Tensor]] = ...) -> Tuple[PackedSequence, Tuple[
                    Tensor, Tensor]]: ...

    def permute_hidden(
        self, hx: Tuple[Tensor, Tensor], permutation: Optional[
                Tensor]) -> Tuple[Tensor, Tensor]: ...

    def check_forward_args(
        self, input: Tensor, hidden: Tuple[Tensor, Tensor],
        batch_sizes: Optional[Tensor]) -> None: ...

    def forward(self, input, hx: Incomplete | None = ...): ...


class QuantizedGRU(QuantizedRNNBase):
    __overloads__: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

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

    def forward(self, input, hx: Incomplete | None = ...): ...


def quantize_rnn_cell_modules(module): ...


def quantize_linear_modules(module, dtype=...): ...


def quantize_rnn_modules(module, dtype=...): ...
