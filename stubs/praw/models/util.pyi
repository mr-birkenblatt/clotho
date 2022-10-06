# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Callable, Generator, List, Optional, Set, Union

from _typeshed import Incomplete


class BoundedSet:
    max_items: Incomplete
    def __init__(self, max_items: int) -> None: ...
    def __contains__(self, item: Any) -> bool: ...
    def add(self, item: Any) -> None: ...


class ExponentialCounter:
    def __init__(self, max_counter: int) -> None: ...
    def counter(self) -> Union[int, float]: ...
    def reset(self) -> None: ...


def permissions_string(
    *, known_permissions: Set[str],
    permissions: Optional[List[str]]) -> str: ...


def stream_generator(
    function: Callable, *, attribute_name: str = ...,
    exclude_before: bool = ..., pause_after: Optional[int] = ...,
    skip_existing: bool = ..., **function_kwargs: Any,
    ) -> Generator[Any, None, None]: ...
