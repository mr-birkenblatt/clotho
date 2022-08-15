# Stubs for pandas.core.groupby.grouper (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument, no-self-use

from typing import Any, Optional


class Grouper:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        ...

    key: Any = ...
    level: Any = ...
    freq: Any = ...
    axis: Any = ...
    sort: Any = ...
    grouper: Any = ...
    obj: Any = ...
    indexer: Any = ...
    binner: Any = ...

    def __init__(
            self, key: Optional[Any] = ..., level: Optional[Any] = ...,
            freq: Optional[Any] = ..., axis: int = ...,
            sort: bool = ...) -> None:
        ...

    @property
    def ax(self) -> Any:
        ...

    @property
    def groups(self) -> Any:
        ...


class Grouping:
    name: Any = ...
    level: Any = ...
    grouper: Any = ...
    all_grouper: Any = ...
    index: Any = ...
    sort: Any = ...
    obj: Any = ...
    observed: Any = ...
    in_axis: Any = ...

    def __init__(
            self, index: Any, grouper: Optional[Any] = ...,
            obj: Optional[Any] = ..., name: Optional[Any] = ...,
            level: Optional[Any] = ..., sort: bool = ...,
            observed: bool = ..., in_axis: bool = ...) -> None:
        ...

    def __iter__(self) -> Any:
        ...

    @property
    def ngroups(self) -> Any:
        ...

    def indices(self) -> Any:
        ...

    @property
    def labels(self) -> Any:
        ...

    def result_index(self) -> Any:
        ...

    @property
    def group_index(self) -> Any:
        ...

    def groups(self) -> Any:
        ...
