from typing import Generic, TypeVar

from _typeshed import Incomplete
from torch._C._distributed_rpc import PyRRef


class AllGatherStates:
    gathered_objects: Incomplete
    proceed_signal: Incomplete
    def __init__(self) -> None: ...

def shutdown(graceful: bool = ..., timeout=...) -> None: ...
def get_worker_info(worker_name: Incomplete | None = ...): ...
T = TypeVar('T')
GenericWithOneTypeVar = Generic[T]

class RRef(PyRRef): ...
class RRefMeta(PyRRef.__class__, GenericWithOneTypeVar.__class__): ...
class RRef(PyRRef, GenericWithOneTypeVar, metaclass=RRefMeta): ...

def method_factory(method_name, docstring): ...

new_method: Incomplete

def remote(to, func, args: Incomplete | None = ..., kwargs: Incomplete | None = ..., timeout=...): ...
def rpc_sync(to, func, args: Incomplete | None = ..., kwargs: Incomplete | None = ..., timeout=...): ...
def rpc_async(to, func, args: Incomplete | None = ..., kwargs: Incomplete | None = ..., timeout=...): ...