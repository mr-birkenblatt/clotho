# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Union

import torch
from _typeshed import Incomplete


def is_built(): ...


class cuFFTPlanCacheAttrContextProp:
    getter: Incomplete
    setter: Incomplete
    def __init__(self, getter, setter) -> None: ...
    def __get__(self, obj, objtype): ...
    def __set__(self, obj, val) -> None: ...


class cuFFTPlanCache:
    device_index: Incomplete
    def __init__(self, device_index) -> None: ...
    size: Incomplete
    max_size: Incomplete
    def clear(self): ...


class cuFFTPlanCacheManager:
    caches: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, device): ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, value): ...


class cuBLASModule:
    def __getattr__(self, name): ...
    def __setattr__(self, name, value): ...


def preferred_linalg_library(
    backend: Union[None, str,
            torch._C._LinalgBackend] = ...) -> torch._C._LinalgBackend: ...


cufft_plan_cache: Incomplete
matmul: Incomplete
