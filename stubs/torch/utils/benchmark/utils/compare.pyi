# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import enum
from typing import List, Optional, Tuple

from _typeshed import Incomplete
from torch.utils.benchmark.utils import common


class Colorize(enum.Enum):
    NONE: str
    COLUMNWISE: str
    ROWWISE: str


class _Column:

    def __init__(
        self, grouped_results: List[Tuple[Optional[common.Measurement],
                ...]], time_scale: float, time_unit: str,
        trim_significant_figures: bool, highlight_warnings: bool) -> None: ...

    def get_results_for(self, group): ...

    def num_to_str(
        self, value: Optional[float], estimated_sigfigs: int,
        spread: Optional[float]): ...


class _Row:

    def __init__(
        self, results, row_group, render_env, env_str_len, row_name_str_len,
        time_scale, colorize,
        num_threads: Incomplete | None = ...) -> None: ...

    def register_columns(self, columns: Tuple[_Column, ...]): ...
    def as_column_strings(self): ...
    @staticmethod
    def color_segment(segment, value, best_value): ...
    def row_separator(self, overall_width): ...
    def finalize_column_strings(self, column_strings, col_widths): ...


class Table:
    results: Incomplete
    label: Incomplete
    row_keys: Incomplete
    column_keys: Incomplete

    def __init__(
        self, results: List[common.Measurement], colorize: Colorize,
        trim_significant_figures: bool, highlight_warnings: bool): ...

    @staticmethod
    def row_fn(m: common.Measurement) -> Tuple[int, Optional[str], str]: ...
    @staticmethod
    def col_fn(m: common.Measurement) -> Optional[str]: ...

    def populate_rows_and_columns(
        self) -> Tuple[Tuple[_Row, ...], Tuple[_Column, ...]]: ...

    def render(self) -> str: ...


class Compare:
    def __init__(self, results: List[common.Measurement]) -> None: ...
    def extend_results(self, results) -> None: ...
    def trim_significant_figures(self) -> None: ...
    def colorize(self, rowwise: bool = ...) -> None: ...
    def highlight_warnings(self) -> None: ...
    def print(self) -> None: ...
