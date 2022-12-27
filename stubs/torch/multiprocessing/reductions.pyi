# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._namedtensor_internals import (
    check_serializing_named_tensor as check_serializing_named_tensor,
)


class StorageWeakRef:
    cdata: Incomplete
    def __init__(self, storage) -> None: ...
    def expired(self): ...
    def __del__(self) -> None: ...


class SharedCache(dict):
    limit: int
    def __init__(self) -> None: ...
    def get(self, key): ...
    def __setitem__(self, key, storage_ref) -> None: ...
    def free_dead_references(self) -> None: ...


shared_cache: Incomplete


def rebuild_event(device, handle): ...


def reduce_event(event): ...


def rebuild_tensor(cls, storage, metadata): ...


def rebuild_cuda_tensor(
    tensor_cls, tensor_size, tensor_stride, tensor_offset, storage_cls,
    dtype, storage_device, storage_handle, storage_size_bytes,
    storage_offset_bytes, requires_grad, ref_counter_handle,
    ref_counter_offset, event_handle, event_sync_required): ...


def reduce_tensor(tensor): ...


def fd_id(fd): ...


def storage_from_cache(cls, key): ...


def rebuild_storage_fd(cls, df, size): ...


def rebuild_storage_filename(
    cls, manager, handle, size, dtype: Incomplete | None = ...): ...


def rebuild_storage_empty(cls): ...


def rebuild_typed_storage(storage, dtype): ...


def reduce_typed_storage(storage): ...


def rebuild_typed_storage_child(storage, storage_type): ...


def reduce_typed_storage_child(storage): ...


def reduce_storage(storage): ...


def init_reductions() -> None: ...
