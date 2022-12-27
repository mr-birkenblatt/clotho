# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Dict, Union

from _typeshed import Incomplete
from torch.types import Device


def caching_allocator_alloc(
    size, device: Union[Device, int] = ...,
    stream: Incomplete | None = ...): ...


def caching_allocator_delete(mem_ptr) -> None: ...


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = ...) -> None: ...


def empty_cache() -> None: ...


def memory_stats(device: Union[Device, int] = ...) -> Dict[str, Any]: ...


def memory_stats_as_nested_dict(
    device: Union[Device, int] = ...) -> Dict[str, Any]: ...


def reset_accumulated_memory_stats(
    device: Union[Device, int] = ...) -> None: ...


def reset_peak_memory_stats(device: Union[Device, int] = ...) -> None: ...


def reset_max_memory_allocated(device: Union[Device, int] = ...) -> None: ...


def reset_max_memory_cached(device: Union[Device, int] = ...) -> None: ...


def memory_allocated(device: Union[Device, int] = ...) -> int: ...


def max_memory_allocated(device: Union[Device, int] = ...) -> int: ...


def memory_reserved(device: Union[Device, int] = ...) -> int: ...


def max_memory_reserved(device: Union[Device, int] = ...) -> int: ...


def memory_cached(device: Union[Device, int] = ...) -> int: ...


def max_memory_cached(device: Union[Device, int] = ...) -> int: ...


def memory_snapshot(): ...


def memory_summary(
    device: Union[Device, int] = ..., abbreviated: bool = ...) -> str: ...


def list_gpu_processes(device: Union[Device, int] = ...) -> str: ...


def mem_get_info(device: Union[Device, int] = ...) -> int: ...
