from typing import Any, Callable

import torch
from _typeshed import Incomplete


class saved_tensors_hooks:
    pack_hook: Incomplete
    unpack_hook: Incomplete
    def __init__(self, pack_hook: Callable[[torch.Tensor], Any], unpack_hook: Callable[[Any], torch.Tensor]) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any): ...

class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory: bool = ...): ...
