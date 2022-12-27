# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any

import torch
from _typeshed import Incomplete


class autocast(torch.amp.autocast_mode.autocast):
    device: str
    fast_dtype: Incomplete

    def __init__(
        self, enabled: bool = ..., dtype: torch.dtype = ...,
        cache_enabled: bool = ...) -> None: ...

    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...
    def __call__(self, func): ...
