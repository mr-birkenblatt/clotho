# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
from typing import Any, Dict, Iterator, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import ClientException as ClientException
from ...util import cachedproperty as cachedproperty
from .base import RedditBase as RedditBase

class RemovalReason(RedditBase):
    STR_FIELD: str
    def __eq__(self, other: Union[str, 'RemovalReason', object]) -> bool: ...
    def __hash__(self) -> int: ...
    id: Incomplete
    subreddit: Incomplete

    def __init__(
        self, reddit: praw.Reddit, subreddit: praw.models.Subreddit,
        id: Optional[str] = ..., reason_id: Optional[str] = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def delete(self) -> None: ...

    def update(
        self, *, message: Optional[str] = ...,
        title: Optional[str] = ...) -> None: ...


class SubredditRemovalReasons:
    def __getitem__(
        self, reason_id: Union[str, int, slice]) -> RemovalReason: ...

    subreddit: Incomplete
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...
    def __iter__(self) -> Iterator[RemovalReason]: ...
    def add(self, *, message: str, title: str) -> RemovalReason: ...
