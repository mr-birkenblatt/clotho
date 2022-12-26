from typing import Any, Optional

from _typeshed import Incomplete
from torch.types import _dtype


def autocast_decorator(autocast_instance, func): ...

class autocast:
    device: Incomplete
    fast_dtype: Incomplete
    def __init__(self, device_type: str, dtype: Optional[_dtype] = ..., enabled: bool = ..., cache_enabled: Optional[bool] = ...) -> None: ...
    prev_cache_enabled: Incomplete
    prev: Incomplete
    prev_fastdtype: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...
    def __call__(self, func): ...
