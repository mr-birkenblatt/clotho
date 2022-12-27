# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import enum

import torch
from _typeshed import Incomplete
from torch._namedtensor_internals import (
    check_serializing_named_tensor as check_serializing_named_tensor,
)
from torch._namedtensor_internals import is_ellipsis as is_ellipsis


        resolve_ellipsis as resolve_ellipsis,
        single_ellipsis_index as single_ellipsis_index,
        unzip_namedshape as unzip_namedshape, update_names as update_names
from torch.overrides import (
    get_default_nowrap_functions as get_default_nowrap_functions,
)


        handle_torch_function as handle_torch_function,
        has_torch_function as has_torch_function,
        has_torch_function_unary as has_torch_function_unary,
        has_torch_function_variadic as has_torch_function_variadic
from typing import Optional, Tuple


class Tensor(torch._C._TensorBase):
    def __deepcopy__(self, memo): ...
    def __reduce_ex__(self, proto): ...
    def storage(self): ...

    def backward(
        self, gradient: Incomplete | None = ...,
        retain_graph: Incomplete | None = ..., create_graph: bool = ...,
        inputs: Incomplete | None = ...): ...

    def register_hook(self, hook): ...
    def reinforce(self, reward): ...
    detach: Incomplete
    detach_: Incomplete
    def is_shared(self): ...
    def share_memory_(self): ...
    def __reversed__(self): ...

    def norm(
        self, p: str = ..., dim: Incomplete | None = ...,
        keepdim: bool = ..., dtype: Incomplete | None = ...): ...

    def solve(self, other): ...
    def lu(self, pivot: bool = ..., get_infos: bool = ...): ...

    def stft(
        self, n_fft: int, hop_length: Optional[int] = ...,
        win_length: Optional[int] = ..., window: Optional[Tensor] = ...,
        center: bool = ..., pad_mode: str = ..., normalized: bool = ...,
        onesided: Optional[bool] = ...,
        return_complex: Optional[bool] = ...): ...

    def istft(
        self, n_fft: int, hop_length: Optional[int] = ...,
        win_length: Optional[int] = ..., window: Optional[Tensor] = ...,
        center: bool = ..., normalized: bool = ...,
        onesided: Optional[bool] = ..., length: Optional[int] = ...,
        return_complex: bool = ...): ...

    def resize(self, *sizes): ...
    def resize_as(self, tensor): ...
    def split(self, split_size, dim: int = ...): ...

    def unique(
        self, sorted: bool = ..., return_inverse: bool = ...,
        return_counts: bool = ..., dim: Incomplete | None = ...): ...

    def unique_consecutive(
        self, return_inverse: bool = ..., return_counts: bool = ...,
        dim: Incomplete | None = ...): ...

    def __rsub__(self, other): ...
    def __rdiv__(self, other): ...
    __rtruediv__: Incomplete
    __itruediv__: Incomplete
    __pow__: Incomplete
    __ipow__: Incomplete
    def __rmod__(self, other): ...
    def __format__(self, format_spec): ...
    def __rpow__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __rlshift__(self, other): ...
    def __rrshift__(self, other): ...
    def __rmatmul__(self, other): ...
    __pos__: Incomplete
    __neg__: Incomplete
    __abs__: Incomplete
    def __len__(self): ...
    def __iter__(self): ...
    def __hash__(self): ...
    def __dir__(self): ...
    __array_priority__: int
    def __array__(self, dtype: Incomplete | None = ...): ...
    def __array_wrap__(self, array): ...
    def __contains__(self, element): ...
    @property
    def __cuda_array_interface__(self): ...
    def storage_type(self): ...
    def refine_names(self, *names): ...
    def align_to(self, *names): ...
    def unflatten(self, dim, sizes): ...
    def rename_(self, *names, **rename_map): ...
    def rename(self, *names, **rename_map): ...
    def to_sparse_coo(self): ...
    @property
    def grad(self): ...
    @grad.setter
    def grad(self, new_grad): ...
    def grad(self): ...

    @classmethod
    def __torch_function__(
        cls, func, types, args=..., kwargs: Incomplete | None = ...): ...

    __torch_dispatch__: Incomplete
    def __dlpack__(self, stream: Incomplete | None = ...): ...
    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]: ...
    __module__: str
