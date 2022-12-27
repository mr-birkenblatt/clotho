# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._C._distributed_c10d import AllToAllOptions as AllToAllOptions

from .constants import default_pg_timeout as default_pg_timeout
from .rendezvous import (
    register_rendezvous_handler as register_rendezvous_handler,
)
from .rendezvous import rendezvous as rendezvous


        AllreduceCoalescedOptions as AllreduceCoalescedOptions,
        AllreduceOptions as AllreduceOptions,
        BarrierOptions as BarrierOptions,
        BroadcastOptions as BroadcastOptions, DebugLevel as DebugLevel,
        GatherOptions as GatherOptions, PrefixStore as PrefixStore,
        ProcessGroup as ProcessGroup, ProcessGroupGloo as ProcessGroupGloo,
        ProcessGroupMPI as ProcessGroupMPI,
        ProcessGroupNCCL as ProcessGroupNCCL, ReduceOp as ReduceOp,
        ReduceOptions as ReduceOptions,
        ReduceScatterOptions as ReduceScatterOptions,
        ScatterOptions as ScatterOptions, Store as Store,
        get_debug_level as get_debug_level
from typing import Optional

from torch._six import string_classes as string_classes


logger: Incomplete
PG_WRAPPER_STORE_PREFIX: str


def supports_complex(reduceOp: ReduceOp) -> bool: ...


class Backend:
    UNDEFINED: str
    GLOO: str
    NCCL: str
    MPI: str
    TCP: str
    def __new__(cls, name: str): ...
    @classmethod
    def register_backend(cls, name, func) -> None: ...


dist_backend = Backend


class _reduce_op:
    __members__: Incomplete
    def __init__(self) -> None: ...
    def __getattribute__(self, key): ...


reduce_op: Incomplete


class group:
    WORLD: Optional[ProcessGroup]


class GroupMember:
    WORLD: Incomplete
    NON_GROUP_MEMBER: Incomplete

STORE_BASED_BARRIER_PREFIX: str


def is_mpi_available(): ...


def is_nccl_available(): ...


def is_gloo_available(): ...


def is_initialized(): ...


def is_torchelastic_launched(): ...


def get_backend(group: Incomplete | None = ...): ...


def init_process_group(
    backend, init_method: Incomplete | None = ..., timeout=...,
    world_size: int = ..., rank: int = ..., store: Incomplete | None = ...,
    group_name: str = ..., pg_options: Incomplete | None = ...) -> None: ...


def destroy_process_group(group: Incomplete | None = ...) -> None: ...


def get_rank(group: Incomplete | None = ...): ...


def get_world_size(group: Incomplete | None = ...): ...


def isend(tensor, dst, group: Incomplete | None = ..., tag: int = ...): ...


def irecv(
    tensor, src: Incomplete | None = ..., group: Incomplete | None = ...,
    tag: int = ...): ...


def send(
    tensor, dst, group: Incomplete | None = ..., tag: int = ...) -> None: ...


def recv(
    tensor, src: Incomplete | None = ..., group: Incomplete | None = ...,
    tag: int = ...): ...


class P2POp:
    op: Incomplete
    tensor: Incomplete
    peer: Incomplete
    group: Incomplete
    tag: Incomplete

    def __init__(
        self, op, tensor, peer, group: Incomplete | None = ...,
        tag: int = ...) -> None: ...

    def __new__(
        cls, op, tensor, peer, group: Incomplete | None = ...,
        tag: int = ...): ...


def batch_isend_irecv(p2p_op_list): ...


def broadcast_multigpu(
    tensor_list, src, group: Incomplete | None = ..., async_op: bool = ...,
    src_tensor: int = ...): ...


def broadcast(
    tensor, src, group: Incomplete | None = ..., async_op: bool = ...): ...


def all_reduce_multigpu(
    tensor_list, op=..., group: Incomplete | None = ...,
    async_op: bool = ...): ...


def all_reduce(
    tensor, op=..., group: Incomplete | None = ..., async_op: bool = ...): ...


def all_reduce_coalesced(
    tensors, op=..., group: Incomplete | None = ..., async_op: bool = ...): ...


def reduce_multigpu(
    tensor_list, dst, op=..., group: Incomplete | None = ...,
    async_op: bool = ..., dst_tensor: int = ...): ...


def reduce(
    tensor, dst, op=..., group: Incomplete | None = ...,
    async_op: bool = ...): ...


def all_gather_multigpu(
    output_tensor_lists, input_tensor_list, group: Incomplete | None = ...,
    async_op: bool = ...): ...


def all_gather_object(
    object_list, obj, group: Incomplete | None = ...) -> None: ...


def gather_object(
    obj, object_gather_list: Incomplete | None = ..., dst: int = ...,
    group: Incomplete | None = ...) -> None: ...


def broadcast_object_list(
    object_list, src: int = ..., group: Incomplete | None = ...,
    device: Incomplete | None = ...) -> None: ...


def scatter_object_list(
    scatter_object_output_list, scatter_object_input_list, src: int = ...,
    group: Incomplete | None = ...) -> None: ...


def all_gather(
    tensor_list, tensor, group: Incomplete | None = ...,
    async_op: bool = ...): ...


def all_gather_coalesced(
    output_tensor_lists, input_tensor_list, group: Incomplete | None = ...,
    async_op: bool = ...): ...


def gather(
    tensor, gather_list: Incomplete | None = ..., dst: int = ...,
    group: Incomplete | None = ..., async_op: bool = ...): ...


def scatter(
    tensor, scatter_list: Incomplete | None = ..., src: int = ...,
    group: Incomplete | None = ..., async_op: bool = ...): ...


def reduce_scatter_multigpu(
    output_tensor_list, input_tensor_lists, op=...,
    group: Incomplete | None = ..., async_op: bool = ...): ...


def reduce_scatter(
    output, input_list, op=..., group: Incomplete | None = ...,
    async_op: bool = ...): ...


def all_to_all_single(
    output, input, output_split_sizes: Incomplete | None = ...,
    input_split_sizes: Incomplete | None = ...,
    group: Incomplete | None = ..., async_op: bool = ...): ...


def all_to_all(
    output_tensor_list, input_tensor_list, group: Incomplete | None = ...,
    async_op: bool = ...): ...


def barrier(
    group=..., async_op: bool = ..., device_ids: Incomplete | None = ...): ...


def monitored_barrier(
    group=..., timeout: Incomplete | None = ...,
    wait_all_ranks: bool = ...): ...


def new_group(
    ranks: Incomplete | None = ..., timeout=...,
    backend: Incomplete | None = ..., pg_options: Incomplete | None = ...): ...


def new_subgroups(
    group_size: Incomplete | None = ..., group: Incomplete | None = ...,
    timeout=..., backend: Incomplete | None = ...,
    pg_options: Incomplete | None = ...): ...


def new_subgroups_by_enumeration(
    ranks_per_subgroup_list, timeout=..., backend: Incomplete | None = ...,
    pg_options: Incomplete | None = ...): ...
