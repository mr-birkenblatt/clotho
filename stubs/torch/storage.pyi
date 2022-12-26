from typing import TypeVar, Union

import torch
from _typeshed import Incomplete
from torch.types import Storage as Storage


HAS_NUMPY: bool
T = TypeVar('T', bound='Union[_StorageBase, _TypedStorage]')

class _StorageBase:
    is_sparse: bool
    is_sparse_csr: bool
    device: torch.device
    def __init__(self, *args, **kwargs) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> None: ...
    def copy_(self, source: T, non_blocking: bool = ...) -> T: ...
    def nbytes(self) -> int: ...
    def size(self) -> int: ...
    def type(self, dtype: str = ..., non_blocking: bool = ...) -> T: ...
    def cuda(self, device: Incomplete | None = ..., non_blocking: bool = ..., **kwargs) -> T: ...
    def element_size(self) -> int: ...
    def get_device(self) -> int: ...
    def data_ptr(self) -> int: ...
    @classmethod
    def from_buffer(cls, *args, **kwargs) -> T: ...
    def resize_(self, size: int): ...
    def is_pinned(self) -> bool: ...
    def is_shared(self) -> bool: ...
    @property
    def is_cuda(self) -> None: ...
    def __iter__(self): ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def __reduce__(self): ...
    def __sizeof__(self): ...
    def clone(self): ...
    def tolist(self): ...
    def cpu(self): ...
    def double(self): ...
    def float(self): ...
    def half(self): ...
    def long(self): ...
    def int(self): ...
    def short(self): ...
    def char(self): ...
    def byte(self): ...
    def bool(self): ...
    def bfloat16(self): ...
    def complex_double(self): ...
    def complex_float(self): ...
    def pin_memory(self): ...
    def share_memory_(self): ...

class _UntypedStorage(torch._C.StorageBase, _StorageBase):
    @property
    def is_cuda(self): ...

class _TypedStorage:
    is_sparse: bool
    dtype: torch.dtype
    def fill_(self, value): ...
    def __new__(cls, *args, wrap_storage: Incomplete | None = ..., dtype: Incomplete | None = ..., device: Incomplete | None = ...): ...
    def __init__(self, *args, device: Incomplete | None = ..., dtype: Incomplete | None = ..., wrap_storage: Incomplete | None = ...) -> None: ...
    @property
    def is_cuda(self): ...
    def __len__(self): ...
    def __setitem__(self, idx, value) -> None: ...
    def __getitem__(self, idx): ...
    def copy_(self, source: T, non_blocking: bool = ...): ...
    def nbytes(self): ...
    def type(self, dtype: str = ..., non_blocking: bool = ...) -> Union[T, str]: ...
    def cuda(self, device: Incomplete | None = ..., non_blocking: bool = ..., **kwargs) -> T: ...
    def element_size(self): ...
    def get_device(self) -> int: ...
    def __iter__(self): ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def __sizeof__(self): ...
    def clone(self): ...
    def tolist(self): ...
    def cpu(self): ...
    def pin_memory(self): ...
    def share_memory_(self): ...
    @property
    def device(self): ...
    def size(self): ...
    def pickle_storage_type(self): ...
    def __reduce__(self): ...
    def data_ptr(self): ...
    def resize_(self, size) -> None: ...
    @classmethod
    def from_buffer(cls, *args, dtype: Incomplete | None = ..., device: Incomplete | None = ..., **kwargs): ...
    def double(self): ...
    def float(self): ...
    def half(self): ...
    def long(self): ...
    def int(self): ...
    def short(self): ...
    def char(self): ...
    def byte(self): ...
    def bool(self): ...
    def bfloat16(self): ...
    def complex_double(self): ...
    def complex_float(self): ...
    @classmethod
    def from_file(cls, filename, shared, size): ...
    def is_pinned(self): ...
    def is_shared(self): ...

class _LegacyStorageMeta(type):
    dtype: torch.dtype
    def __instancecheck__(cls, instance): ...

class _LegacyStorage(_TypedStorage, metaclass=_LegacyStorageMeta): ...