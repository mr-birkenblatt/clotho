from typing import NamedTuple

from _typeshed import Incomplete


class BackendValue(NamedTuple):
    construct_rpc_backend_options_handler: Incomplete
    init_backend_handler: Incomplete
BackendType: Incomplete

def backend_registered(backend_name): ...
def register_backend(backend_name, construct_rpc_backend_options_handler, init_backend_handler): ...
def construct_rpc_backend_options(backend, rpc_timeout=..., init_method=..., **kwargs): ...
def init_backend(backend, *args, **kwargs): ...

# Names in __all__ with no definition:
#   BackendValue
