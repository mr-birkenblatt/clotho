# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import enum
from typing import (
    Any,
    Callable,
    Dict,
    NoReturn,
    Optional,
    overload,
    Tuple,
    Union,
)

from _typeshed import Incomplete
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.valgrind_wrapper import (
    timer_interface as valgrind_timer_interface,
)


def timer() -> float: ...


timer: Incomplete


class Language(enum.Enum):
    PYTHON: int
    CPP: int


class CPPTimer:

    def __init__(
        self, stmt: str, setup: str, global_setup: str, timer: Callable[[],
                float], globals: Dict[str, Any]) -> None: ...

    def timeit(self, number: int) -> float: ...


class Timer:

    def __init__(
        self, stmt: str = ..., setup: str = ..., global_setup: str = ...,
        timer: Callable[[], float] = ..., globals: Optional[Dict[str,
                        Any]] = ..., label: Optional[str] = ...,
        sub_label: Optional[str] = ..., description: Optional[str] = ...,
        env: Optional[str] = ..., num_threads: int = ...,
        language: Union[Language, str] = ...) -> None: ...

    def timeit(self, number: int = ...) -> common.Measurement: ...
    def repeat(self, repeat: int = ..., number: int = ...) -> None: ...

    def autorange(
        self, callback: Optional[Callable[[
                                int, float], NoReturn]] = ...) -> None: ...

    def adaptive_autorange(
        self, threshold: float = ..., *, min_run_time: float = ...,
        max_run_time: float = ..., callback: Optional[Callable[[int, float],
                        NoReturn]] = ...) -> common.Measurement: ...

    def blocked_autorange(
        self, callback: Optional[Callable[[int, float], NoReturn]] = ...,
        min_run_time: float = ...) -> common.Measurement: ...

    @overload
    def collect_callgrind(
        self, number: int, *, repeats: None, collect_baseline: bool,
        retain_out_file: bool) -> valgrind_timer_interface.CallgrindStats: ...

    @overload
    def collect_callgrind(
        self, number: int, *, repeats: int, collect_baseline: bool,
        retain_out_file: bool) -> Tuple[
            valgrind_timer_interface.CallgrindStats, ...]: ...
