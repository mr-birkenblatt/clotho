# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


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
