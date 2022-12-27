# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Generator, Tuple
from urllib.parse import urlparse as urlparse

from _typeshed import Incomplete
from torch._C._distributed_c10d import Store as Store
from torch._C._distributed_rpc import (
    enable_gil_profiling as enable_gil_profiling,
)
from torch._C._distributed_rpc import get_rpc_timeout as get_rpc_timeout
from torch._C._distributed_rpc import PyRRef as PyRRef
from torch._C._distributed_rpc import (
    RemoteProfilerManager as RemoteProfilerManager,
)
from torch._C._distributed_rpc import RpcAgent as RpcAgent
from torch._C._distributed_rpc import RpcBackendOptions as RpcBackendOptions
from torch._C._distributed_rpc import TensorPipeAgent as TensorPipeAgent
from torch._C._distributed_rpc import WorkerInfo as WorkerInfo

from . import api as api
from . import backend_registry as backend_registry
from . import functions as functions
from .api import *
from .backend_registry import BackendType as BackendType
from .options import TensorPipeRpcBackendOptions as TensorPipeRpcBackendOptions


logger: Incomplete


def is_available(): ...


rendezvous_iterator: Generator[Tuple[Store, int, int], None, None]


def init_rpc(
    name, backend: Incomplete | None = ..., rank: int = ...,
        world_size: Incomplete | None = ...,
        rpc_backend_options: Incomplete | None = ...) -> None: ...
