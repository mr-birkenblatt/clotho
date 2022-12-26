from _typeshed import Incomplete
from typing_extensions import Protocol

from ..parameter import is_lazy as is_lazy


class _LazyProtocol(Protocol):
    def register_forward_pre_hook(self, hook) -> None: ...

class LazyModuleMixin:
    cls_to_become: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def initialize_parameters(self, *args, **kwargs): ...
    def has_uninitialized_params(self): ...
