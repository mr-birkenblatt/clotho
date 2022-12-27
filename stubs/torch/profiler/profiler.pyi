# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from enum import Enum
from typing import Any, Callable, Iterable, Optional

from _typeshed import Incomplete
from torch._C._autograd import _ExperimentalConfig
from torch.autograd import kineto_available as kineto_available
from torch.autograd import ProfilerActivity as ProfilerActivity


def supported_activities(): ...


class _KinetoProfile:
    activities: Incomplete
    record_shapes: Incomplete
    with_flops: Incomplete
    profile_memory: Incomplete
    with_stack: Incomplete
    with_modules: Incomplete
    experimental_config: Incomplete
    profiler: Incomplete

    def __init__(
        self, *, activities: Optional[Iterable[ProfilerActivity]] = ...,
        record_shapes: bool = ..., profile_memory: bool = ...,
        with_stack: bool = ..., with_flops: bool = ...,
        with_modules: bool = ...,
        experimental_config: Optional[_ExperimentalConfig] = ...) -> None: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def prepare_trace(self) -> None: ...
    def start_trace(self) -> None: ...
    def stop_trace(self) -> None: ...
    def export_chrome_trace(self, path: str): ...
    def export_stacks(self, path: str, metric: str = ...): ...

    def key_averages(
        self, group_by_input_shape: bool = ...,
        group_by_stack_n: int = ...): ...

    def events(self): ...
    def add_metadata(self, key: str, value: str): ...
    def add_metadata_json(self, key: str, value: str): ...


class ProfilerAction(Enum):
    NONE: int
    WARMUP: int
    RECORD: int
    RECORD_AND_SAVE: int


def schedule(
    *, wait: int, warmup: int, active: int, repeat: int = ...,
        skip_first: int = ...) -> Callable: ...


def tensorboard_trace_handler(
    dir_name: str, worker_name: Optional[str] = ..., use_gzip: bool = ...): ...


class profile(_KinetoProfile):
    schedule: Incomplete
    record_steps: bool
    on_trace_ready: Incomplete
    step_num: int
    current_action: Incomplete
    step_rec_fn: Incomplete
    action_map: Incomplete

    def __init__(
        self, *, activities: Optional[Iterable[ProfilerActivity]] = ...,
        schedule: Optional[Callable[[int], ProfilerAction]] = ...,
        on_trace_ready: Optional[Callable[..., Any]] = ...,
        record_shapes: bool = ..., profile_memory: bool = ...,
        with_stack: bool = ..., with_flops: bool = ...,
        with_modules: bool = ...,
        experimental_config: Optional[_ExperimentalConfig] = ...,
        use_cuda: Optional[bool] = ...) -> None: ...

    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def step(self) -> None: ...
