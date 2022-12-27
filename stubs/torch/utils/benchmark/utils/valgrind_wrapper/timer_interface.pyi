# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import enum
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

from _typeshed import Incomplete
from torch.utils.benchmark.utils import common


FunctionCount: Incomplete


class FunctionCounts:
    inclusive: bool
    truncate_rows: bool
    def __iter__(self) -> Generator[FunctionCount, None, None]: ...
    def __len__(self) -> int: ...

    def __getitem__(
        self, item: Any) -> Union[FunctionCount, 'FunctionCounts']: ...

    def __add__(self, other: FunctionCounts) -> FunctionCounts: ...
    def __sub__(self, other: FunctionCounts) -> FunctionCounts: ...
    def __mul__(self, other: Union[int, float]) -> FunctionCounts: ...
    def transform(self, map_fn: Callable[[str], str]) -> FunctionCounts: ...
    def filter(self, filter_fn: Callable[[str], bool]) -> FunctionCounts: ...
    def sum(self) -> int: ...
    def denoise(self) -> FunctionCounts: ...

    def __init__(
        self, _data, inclusive, truncate_rows, _linewidth) -> None: ...


class CallgrindStats:
    task_spec: common.TaskSpec
    number_per_run: int
    built_with_debug_symbols: bool
    baseline_inclusive_stats: FunctionCounts
    baseline_exclusive_stats: FunctionCounts
    stmt_inclusive_stats: FunctionCounts
    stmt_exclusive_stats: FunctionCounts
    stmt_callgrind_out: Optional[str]
    def stats(self, inclusive: bool = ...) -> FunctionCounts: ...
    def counts(self, *, denoise: bool = ...) -> int: ...

    def delta(
        self, other: CallgrindStats,
        inclusive: bool = ...) -> FunctionCounts: ...

    def as_standardized(self) -> CallgrindStats: ...

    def __init__(
        self, task_spec, number_per_run, built_with_debug_symbols,
        baseline_inclusive_stats, baseline_exclusive_stats,
        stmt_inclusive_stats, stmt_exclusive_stats,
        stmt_callgrind_out) -> None: ...


class Serialization(enum.Enum):
    PICKLE: int
    TORCH: int
    TORCH_JIT: int


class CopyIfCallgrind:
    def __init__(self, value: Any, *, setup: Optional[str] = ...) -> None: ...
    @property
    def value(self) -> Any: ...
    @property
    def setup(self) -> Optional[str]: ...
    @property
    def serialization(self) -> Serialization: ...
    @staticmethod
    def unwrap_all(globals: Dict[str, Any]) -> Dict[str, Any]: ...


class GlobalsBridge:
    def __init__(self, globals: Dict[str, Any], data_dir: str) -> None: ...
    def construct(self) -> str: ...


class _ValgrindWrapper:
    def __init__(self) -> None: ...

    def collect_callgrind(
        self, task_spec: common.TaskSpec, globals: Dict[str, Any], *,
        number: int, repeats: int, collect_baseline: bool, is_python: bool,
        retain_out_file: bool) -> Tuple[CallgrindStats, ...]: ...
