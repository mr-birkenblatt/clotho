# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Optional

from _typeshed import Incomplete
from torch.types import _dtype


def autocast_decorator(autocast_instance, func): ...


class autocast:
    device: Incomplete
    fast_dtype: Incomplete

    def __init__(
        self, device_type: str, dtype: Optional[_dtype] = ...,
        enabled: bool = ..., cache_enabled: Optional[bool] = ...) -> None: ...

    prev_cache_enabled: Incomplete
    prev: Incomplete
    prev_fastdtype: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...
    def __call__(self, func): ...
