# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from collections import defaultdict
from collections.abc import Generator
from typing import NamedTuple

from _typeshed import Incomplete
from torch.autograd import DeviceType as DeviceType


class EventList(list):
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def self_cpu_time_total(self): ...

    def table(
        self, sort_by: Incomplete | None = ..., row_limit: int = ...,
        max_src_column_width: int = ..., header: Incomplete | None = ...,
        top_level_events_only: bool = ...): ...

    def export_chrome_trace(self, path) -> None: ...
    def supported_export_stacks_metrics(self): ...
    def export_stacks(self, path: str, metric: str): ...

    def key_averages(
        self, group_by_input_shapes: bool = ...,
        group_by_stack_n: int = ...): ...

    def total_average(self): ...


class FormattedTimesMixin:
    cpu_time_str: Incomplete
    cuda_time_str: Incomplete
    cpu_time_total_str: Incomplete
    cuda_time_total_str: Incomplete
    self_cpu_time_total_str: Incomplete
    self_cuda_time_total_str: Incomplete
    @property
    def cpu_time(self): ...
    @property
    def cuda_time(self): ...


class Interval:
    start: Incomplete
    end: Incomplete
    def __init__(self, start, end) -> None: ...
    def elapsed_us(self): ...


class Kernel(NamedTuple):
    name: Incomplete
    device: Incomplete
    duration: Incomplete


class FunctionEvent(FormattedTimesMixin):
    id: Incomplete
    node_id: Incomplete
    name: Incomplete
    trace_name: Incomplete
    time_range: Incomplete
    thread: Incomplete
    fwd_thread: Incomplete
    kernels: Incomplete
    count: int
    cpu_children: Incomplete
    cpu_parent: Incomplete
    input_shapes: Incomplete
    stack: Incomplete
    scope: Incomplete
    cpu_memory_usage: Incomplete
    cuda_memory_usage: Incomplete
    is_async: Incomplete
    is_remote: Incomplete
    sequence_nr: Incomplete
    device_type: Incomplete
    device_index: Incomplete
    is_legacy: Incomplete
    flops: Incomplete

    def __init__(
        self, id, name, thread, start_us, end_us,
        fwd_thread: Incomplete | None = ...,
        input_shapes: Incomplete | None = ...,
        stack: Incomplete | None = ..., scope: int = ...,
        cpu_memory_usage: int = ..., cuda_memory_usage: int = ...,
        is_async: bool = ..., is_remote: bool = ..., sequence_nr: int = ...,
        node_id: int = ..., device_type=..., device_index: int = ...,
        is_legacy: bool = ..., flops: Incomplete | None = ...,
        trace_name: Incomplete | None = ...) -> None: ...

    def append_kernel(self, name, device, duration) -> None: ...
    def append_cpu_child(self, child) -> None: ...
    def set_cpu_parent(self, parent) -> None: ...
    @property
    def self_cpu_memory_usage(self): ...
    @property
    def self_cuda_memory_usage(self): ...
    @property
    def self_cpu_time_total(self): ...
    @property
    def cuda_time_total(self): ...
    @property
    def self_cuda_time_total(self): ...
    @property
    def cpu_time_total(self): ...
    @property
    def key(self): ...


class FunctionEventAvg(FormattedTimesMixin):
    key: Incomplete
    count: int
    node_id: int
    is_async: bool
    is_remote: bool
    cpu_time_total: int
    cuda_time_total: int
    self_cpu_time_total: int
    self_cuda_time_total: int
    input_shapes: Incomplete
    stack: Incomplete
    scope: Incomplete
    cpu_memory_usage: int
    cuda_memory_usage: int
    self_cpu_memory_usage: int
    self_cuda_memory_usage: int
    cpu_children: Incomplete
    cpu_parent: Incomplete
    device_type: Incomplete
    is_legacy: bool
    flops: int
    def __init__(self) -> None: ...
    def add(self, other): ...
    def __iadd__(self, other): ...


class StringTable(defaultdict):
    def __missing__(self, key): ...


class MemRecordsAcc:
    def __init__(self, mem_records) -> None: ...

    def in_interval(
        self, start_us, end_us) -> Generator[Incomplete, None, None]: ...


MEMORY_EVENT_NAME: str
