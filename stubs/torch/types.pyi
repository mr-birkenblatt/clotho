# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import builtins
from typing import Union

import torch
from _typeshed import Incomplete


class SymInt:
    ...


Number = Union[builtins.int, builtins.float, builtins.bool]
Device: Incomplete


class Storage:
    device: torch.device
    dtype: torch.dtype
    def __deepcopy__(self, memo) -> Storage: ...
    def element_size(self) -> int: ...
    def is_shared(self) -> bool: ...
    def share_memory_(self) -> Storage: ...
    def nbytes(self) -> int: ...
    def cpu(self) -> Storage: ...
    def data_ptr(self) -> int: ...

    def from_file(
        self, filename: str, shared: bool = ...,
        nbytes: int = ...) -> Storage: ...
