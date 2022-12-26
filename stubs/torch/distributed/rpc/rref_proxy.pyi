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
