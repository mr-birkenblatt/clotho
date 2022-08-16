# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Iterator

import praw
from _typeshed import Incomplete

from ..base import PRAWBase as PRAWBase

class BaseList(PRAWBase):
    CHILD_ATTRIBUTE: Incomplete
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...
    def __contains__(self, item: Any) -> bool: ...
    def __getitem__(self, index: int) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...
