# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Generator, List, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import ClientException as ClientException
from ...util.cache import cachedproperty as cachedproperty
from ..base import PRAWBase as PRAWBase
from .base import RedditBase as RedditBase
from .submission import Submission as Submission
from .subreddit import Subreddit as Subreddit

class CollectionModeration(PRAWBase):
    collection_id: Incomplete
    def __init__(self, reddit: praw.Reddit, collection_id: str) -> None: ...
    def add_post(self, submission: praw.models.Submission) -> None: ...
    def delete(self) -> None: ...
    def remove_post(self, submission: praw.models.Submission) -> None: ...

    def reorder(
        self, links: List[Union[str, 'praw.models.Submission']]) -> None: ...

    def update_description(self, description: str) -> None: ...
    def update_display_layout(self, display_layout: str) -> None: ...
    def update_title(self, title: str) -> None: ...


class Collection(RedditBase):
    STR_FIELD: str
    def mod(self) -> CollectionModeration: ...
    def subreddit(self) -> praw.models.Subreddit: ...

    collection_id: Incomplete

    def __init__(
        self, reddit: praw.Reddit, _data: Dict[str, Any] = ...,
        collection_id: Optional[str] = ...,
        permalink: Optional[str] = ...) -> None: ...

    def __iter__(self) -> Generator[Any, None, None]: ...
    def __len__(self) -> int: ...
    author: Incomplete
    def __setattr__(self, attribute: str, value: Any) -> None: ...
    def follow(self) -> None: ...
    def unfollow(self) -> None: ...


class SubredditCollectionsModeration(PRAWBase):
    subreddit_fullname: Incomplete

    def __init__(
        self, reddit: praw.Reddit, sub_fullname: str,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def create(
        self, *, description: str, display_layout: Optional[str] = ...,
        title: str) -> None: ...


class SubredditCollections(PRAWBase):
    def mod(self) -> SubredditCollectionsModeration: ...

    def __call__(
        self, collection_id: Optional[str] = ...,
        permalink: Optional[str] = ...) -> None: ...

    subreddit: Incomplete

    def __init__(
        self, reddit: praw.Reddit, subreddit: praw.models.Subreddit,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def __iter__(self) -> None: ...
