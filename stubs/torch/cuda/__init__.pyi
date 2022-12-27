# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch._C

from . import amp as amp
from . import jiterator as jiterator
from . import nvtx as nvtx
from .memory import *
from .random import *


        profiler as profiler, sparse as sparse
from .graphs import CUDAGraph as CUDAGraph
from .graphs import graph as graph


        graph_pool_handle as graph_pool_handle,
        is_current_stream_capturing as is_current_stream_capturing,
        make_graphed_callables as make_graphed_callables
from .streams import Event as Event
from .streams import ExternalStream as ExternalStream


        Stream as Stream
from typing import Any, List, Optional, Tuple, Union

from _typeshed import Incomplete
from torch.storage import _LegacyStorage
from torch.types import Device as Device


class _LazySeedTracker:
    manual_seed_all_cb: Incomplete
    manual_seed_cb: Incomplete
    call_order: Incomplete
    def __init__(self) -> None: ...
    def queue_seed_all(self, cb, traceback) -> None: ...
    def queue_seed(self, cb, traceback) -> None: ...
    def get_calls(self) -> List: ...


has_magma: bool
has_half: bool
default_generators: Tuple[torch._C.Generator]


def is_available() -> bool: ...


def is_bf16_supported(): ...


def is_initialized(): ...


class DeferredCudaCallError(Exception):
    ...


def init() -> None: ...


def cudart(): ...


class cudaStatus:
    SUCCESS: int
    ERROR_NOT_READY: int


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None: ...


def check_error(res: int) -> None: ...


class device:
    idx: Incomplete
    prev_idx: int
    def __init__(self, device: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...


class device_of(device):
    def __init__(self, obj) -> None: ...


def set_device(device: _device_t) -> None: ...


def get_device_name(device: Optional[_device_t] = ...) -> str: ...


def get_device_capability(
    device: Optional[_device_t] = ...) -> Tuple[int, int]: ...


def get_device_properties(device: _device_t) -> _CudaDeviceProperties: ...


def can_device_access_peer(
    device: _device_t, peer_device: _device_t) -> bool: ...


class StreamContext:
    cur_stream: Optional['torch.cuda.Stream']
    stream: Incomplete
    idx: Incomplete
    src_prev_stream: Incomplete
    dst_prev_stream: Incomplete
    def __init__(self, stream: Optional['torch.cuda.Stream']) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any): ...


def stream(stream: Optional['torch.cuda.Stream']) -> StreamContext: ...


def set_stream(stream: Stream): ...


def device_count() -> int: ...


def get_arch_list() -> List[str]: ...


def get_gencode_flags() -> str: ...


def current_device() -> int: ...


def synchronize(device: _device_t = ...) -> None: ...


def ipc_collect(): ...


def current_stream(device: Optional[_device_t] = ...) -> Stream: ...


def default_stream(device: Optional[_device_t] = ...) -> Stream: ...


def current_blas_handle(): ...


def set_sync_debug_mode(debug_mode: Union[int, str]) -> None: ...


def get_sync_debug_mode() -> int: ...


def memory_usage(device: Optional[Union[Device, int]] = ...) -> int: ...


def utilization(device: Optional[Union[Device, int]] = ...) -> int: ...


class _CudaBase:
    is_cuda: bool
    is_sparse: bool
    def type(self, *args, **kwargs): ...
    __new__: Incomplete


class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs) -> None: ...


class ByteStorage(_CudaLegacyStorage):
    def dtype(self): ...


class DoubleStorage(_CudaLegacyStorage):
    def dtype(self): ...


class FloatStorage(_CudaLegacyStorage):
    def dtype(self): ...


class HalfStorage(_CudaLegacyStorage):
    def dtype(self): ...


class LongStorage(_CudaLegacyStorage):
    def dtype(self): ...


class IntStorage(_CudaLegacyStorage):
    def dtype(self): ...


class ShortStorage(_CudaLegacyStorage):
    def dtype(self): ...


class CharStorage(_CudaLegacyStorage):
    def dtype(self): ...


class BoolStorage(_CudaLegacyStorage):
    def dtype(self): ...


class BFloat16Storage(_CudaLegacyStorage):
    def dtype(self): ...


class ComplexDoubleStorage(_CudaLegacyStorage):
    def dtype(self): ...


class ComplexFloatStorage(_CudaLegacyStorage):
    def dtype(self): ...
