# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Dict, Iterable, List, Optional


class TaskSpec:
    stmt: str
    setup: str
    global_setup: str
    label: Optional[str]
    sub_label: Optional[str]
    description: Optional[str]
    env: Optional[str]
    num_threads: int
    @property
    def title(self) -> str: ...
    def setup_str(self) -> str: ...
    def summarize(self) -> str: ...

    def __init__(
        self, stmt, setup, global_setup, label, sub_label, description, env,
        num_threads) -> None: ...


class Measurement:
    number_per_run: int
    raw_times: List[float]
    task_spec: TaskSpec
    metadata: Optional[Dict[Any, Any]]
    def __post_init__(self) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    @property
    def times(self) -> List[float]: ...
    @property
    def median(self) -> float: ...
    @property
    def mean(self) -> float: ...
    @property
    def iqr(self) -> float: ...
    @property
    def significant_figures(self) -> int: ...
    @property
    def has_warnings(self) -> bool: ...
    def meets_confidence(self, threshold: float = ...) -> bool: ...
    @property
    def title(self) -> str: ...
    @property
    def env(self) -> str: ...
    @property
    def as_row_name(self) -> str: ...

    @staticmethod
    def merge(
        measurements: Iterable['Measurement']) -> List['Measurement']: ...

    def __init__(
        self, number_per_run, raw_times, task_spec, metadata) -> None: ...


# Names in __all__ with no definition:
#   _make_temp_dir
