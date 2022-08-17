# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, List, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import ClientException as ClientException
from .base import RedditBase as RedditBase

class Emoji(RedditBase):
    STR_FIELD: str

    def __eq__(self, other: Union[str, 'Emoji']) -> bool: ...
    def __hash__(self) -> int: ...

    name: Incomplete
    subreddit: Incomplete

    def __init__(
        self, reddit: praw.Reddit, subreddit: praw.models.Subreddit,
        name: str, _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def delete(self) -> None: ...

    def update(
        self, *, mod_flair_only: Optional[bool] = ...,
        post_flair_allowed: Optional[bool] = ...,
        user_flair_allowed: Optional[bool] = ...) -> None: ...


class SubredditEmoji:
    def __getitem__(self, name: str) -> Emoji: ...

    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...
    def __iter__(self) -> List[Emoji]: ...

    def add(
        self, *, image_path: str, mod_flair_only: Optional[bool] = ...,
        name: str, post_flair_allowed: Optional[bool] = ...,
        user_flair_allowed: Optional[bool] = ...) -> Emoji: ...
