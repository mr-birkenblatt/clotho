from typing import Iterable, Union

from _typeshed import Incomplete


GlobPattern = Union[str, Iterable[str]]

class GlobGroup:
    include: Incomplete
    exclude: Incomplete
    separator: Incomplete
    def __init__(self, include: GlobPattern, *, exclude: GlobPattern = ..., separator: str = ...) -> None: ...
    def matches(self, candidate: str) -> bool: ...
