# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.futures import Future as Future

from . import functions as functions
from . import rpc_async as rpc_async
from .constants import UNSET_RPC_TIMEOUT as UNSET_RPC_TIMEOUT


class RRefProxy:
    rref: Incomplete
    rpc_api: Incomplete
    rpc_timeout: Incomplete
    def __init__(self, rref, rpc_api, timeout=...) -> None: ...
    def __getattr__(self, func_name): ...
