from enum import Enum
from typing import NamedTuple

from _typeshed import Incomplete


class RPCExecMode(Enum):
    SYNC: str
    ASYNC: str
    ASYNC_JIT: str
    REMOTE: str

class _InternalRPCPickler:
    def __init__(self) -> None: ...
    def serialize(self, obj): ...
    def deserialize(self, binary_data, tensor_table): ...

def serialize(obj): ...
def deserialize(binary_data, tensor_table): ...

class PythonUDF(NamedTuple):
    func: Incomplete
    args: Incomplete
    kwargs: Incomplete

class RemoteException(NamedTuple):
    msg: Incomplete
    exception_type: Incomplete
