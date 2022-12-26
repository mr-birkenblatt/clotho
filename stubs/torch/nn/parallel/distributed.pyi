from collections.abc import Generator
from enum import Enum
from typing import Any, Callable

from _typeshed import Incomplete
from torch.autograd import Function as Function
from torch.autograd import Variable as Variable
from torch.distributed.algorithms.join import Join as Join
from torch.distributed.algorithms.join import Joinable as Joinable
from torch.distributed.algorithms.join import JoinHook as JoinHook
from torch.distributed.distributed_c10d import ReduceOp as ReduceOp
from torch.distributed.rpc import RRef as RRef
from torch.utils._pytree import tree_flatten as tree_flatten
from torch.utils._pytree import tree_unflatten as tree_unflatten

from ..modules import Module as Module
from .scatter_gather import gather as gather
from .scatter_gather import is_namedtuple as is_namedtuple
from .scatter_gather import scatter_kwargs as scatter_kwargs


RPC_AVAILABLE: bool
logger: Incomplete

class _BufferCommHookLocation(Enum):
    PRE_FORWARD: Incomplete
    POST_FORWARD: Incomplete

class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation
    def __init__(self, buffer_comm_hook, buffer_comm_hook_state, buffer_comm_hook_location) -> None: ...

class _DDPSink(Function):
    @staticmethod
    def forward(ctx, reducer, state_dict, *inputs): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class _DDPJoinHook(JoinHook):
    ddp: Incomplete
    def __init__(self, ddp, divide_by_initial_world_size) -> None: ...
    def main_hook(self) -> None: ...
    def post_hook(self, is_last_joiner: bool): ...

class DistributedDataParallel(Module, Joinable):
    logger: Incomplete
    is_multi_device_module: Incomplete
    device_type: Incomplete
    device_ids: Incomplete
    output_device: Incomplete
    process_group: Incomplete
    static_graph: bool
    dim: Incomplete
    module: Incomplete
    device: Incomplete
    broadcast_buffers: Incomplete
    find_unused_parameters: Incomplete
    require_backward_grad_sync: bool
    require_forward_param_sync: bool
    gradient_as_bucket_view: Incomplete
    parameters_to_ignore: Incomplete
    broadcast_bucket_size: Incomplete
    bucket_bytes_cap: Incomplete
    use_side_stream_for_tensor_copies: Incomplete
    def __init__(self, module, device_ids: Incomplete | None = ..., output_device: Incomplete | None = ..., dim: int = ..., broadcast_buffers: bool = ..., process_group: Incomplete | None = ..., bucket_cap_mb: int = ..., find_unused_parameters: bool = ..., check_reduction: bool = ..., gradient_as_bucket_view: bool = ..., static_graph: bool = ...) -> None: ...
    def no_sync(self) -> Generator[None, None, None]: ...
    def forward(self, *inputs, **kwargs): ...
    def scatter(self, inputs, kwargs, device_ids): ...
    def to_kwargs(self, inputs, kwargs, device_id): ...
    def gather(self, outputs, output_device): ...
    def train(self, mode: bool = ...): ...
    def join(self, divide_by_initial_world_size: bool = ..., enable: bool = ..., throw_on_early_termination: bool = ...): ...
    def join_hook(self, **kwargs): ...
    @property
    def join_device(self): ...
    @property
    def join_process_group(self): ...
    def register_comm_hook(self, state: object, hook: callable): ...
    def will_sync_module_buffers(self): ...
