# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, NamedTuple, Tuple, Union

from _typeshed import Incomplete
from torch import Tensor as Tensor

from ..._jit_internal import Optional as Optional


class PackedSequence_(NamedTuple):
    data: Incomplete
    batch_sizes: Incomplete
    sorted_indices: Incomplete
    unsorted_indices: Incomplete


def bind(optional, fn): ...


class PackedSequence(PackedSequence_):

    def __new__(
        cls, data, batch_sizes: Incomplete | None = ...,
        sorted_indices: Incomplete | None = ...,
        unsorted_indices: Incomplete | None = ...): ...

    def pin_memory(self): ...
    def cuda(self, *args, **kwargs): ...
    def cpu(self, *args, **kwargs): ...
    def double(self): ...
    def float(self): ...
    def half(self): ...
    def long(self): ...
    def int(self): ...
    def short(self): ...
    def char(self): ...
    def byte(self): ...
    def to(self, *args, **kwargs): ...
    @property
    def is_cuda(self): ...
    def is_pinned(self): ...


def invert_permutation(permutation: Optional[Tensor]) -> Optional[Tensor]: ...


def pack_padded_sequence(
    input: Tensor, lengths: Tensor, batch_first: bool = ...,
    enforce_sorted: bool = ...) -> PackedSequence: ...


def pad_packed_sequence(
    sequence: PackedSequence, batch_first: bool = ...,
    padding_value: float = ..., total_length: Optional[int] = ...) -> Tuple[
        Tensor, Tensor]: ...


def pad_sequence(
    sequences: Union[Tensor, List[Tensor]], batch_first: bool = ...,
    padding_value: float = ...) -> Tensor: ...


def unpad_sequence(
    padded_sequences: Tensor, lengths: Tensor,
    batch_first: bool = ...) -> List[Tensor]: ...


def pack_sequence(
    sequences: List[Tensor], enforce_sorted: bool = ...) -> PackedSequence: ...


def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]: ...
