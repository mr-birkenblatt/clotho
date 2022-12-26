from torch._C._distributed_rpc_testing import (
    FaultyTensorPipeAgent as FaultyTensorPipeAgent,
)
from torch._C._distributed_rpc_testing import (
    FaultyTensorPipeRpcBackendOptions as FaultyTensorPipeRpcBackendOptions,
)

from . import faulty_agent_backend_registry as faulty_agent_backend_registry


def is_available(): ...
