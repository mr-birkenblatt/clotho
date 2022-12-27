# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from contextlib import ContextDecorator

from _typeshed import Incomplete
from torch.autograd import DeviceType as DeviceType


        ProfilerActivity as ProfilerActivity,
        ProfilerConfig as ProfilerConfig, ProfilerState as ProfilerState,
        kineto_available as kineto_available
from torch.autograd.profiler_util import EventList as EventList


        FunctionEvent as FunctionEvent,
        MEMORY_EVENT_NAME as MEMORY_EVENT_NAME, MemRecordsAcc as MemRecordsAcc
from typing import Any, Optional

from torch.futures import Future as Future


class ContextDecorator:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def __call__(self, func): ...


class profile:
    enabled: Incomplete
    use_cuda: Incomplete
    function_events: Incomplete
    entered: bool
    record_shapes: Incomplete
    with_flops: Incomplete
    profile_memory: Incomplete
    with_stack: Incomplete
    with_modules: Incomplete
    use_cpu: Incomplete
    experimental_config: Incomplete
    kineto_results: Incomplete
    kineto_activities: Incomplete
    profiler_kind: Incomplete

    def __init__(
        self, enabled: bool = ..., *, use_cuda: bool = ...,
        record_shapes: bool = ..., with_flops: bool = ...,
        profile_memory: bool = ..., with_stack: bool = ...,
        with_modules: bool = ..., use_kineto: bool = ...,
        use_cpu: bool = ...,
        experimental_config: Incomplete | None = ...) -> None: ...

    def config(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

    def table(
        self, sort_by: Incomplete | None = ..., row_limit: int = ...,
        max_src_column_width: int = ..., header: Incomplete | None = ...,
        top_level_events_only: bool = ...): ...

    def export_chrome_trace(self, path): ...
    def export_stacks(self, path: str, metric: str = ...): ...

    def key_averages(
        self, group_by_input_shape: bool = ...,
        group_by_stack_n: int = ...): ...

    def total_average(self): ...
    @property
    def self_cpu_time_total(self): ...


class record_function(ContextDecorator):
    name: Incomplete
    args: Incomplete
    run_callbacks_on_exit: bool
    handle: Incomplete
    def __init__(self, name: str, args: Optional[str] = ...) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any): ...


class emit_nvtx:
    enabled: Incomplete
    entered: bool
    record_shapes: Incomplete

    def __init__(
        self, enabled: bool = ..., record_shapes: bool = ...) -> None: ...

    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...


def load_nvprof(path): ...


class EnforceUnique:
    seen: Incomplete
    def __init__(self) -> None: ...
    def see(self, *key) -> None: ...


def parse_nvprof_trace(path): ...


def kineto_step() -> None: ...
