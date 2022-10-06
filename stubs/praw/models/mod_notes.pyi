# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,super-init-not-called
from typing import Any, Generator, List, Optional, Tuple, Union

import praw
from _typeshed import Incomplete

from ..const import API_PATH as API_PATH
from ..models.base import PRAWBase as PRAWBase
from ..models.listing.generator import ListingGenerator as ListingGenerator
from .reddit.comment import Comment as Comment
from .reddit.redditor import Redditor as Redditor
from .reddit.submission import Submission as Submission


RedditorType = Union[Redditor, str]
SubredditType: Incomplete = ...
ThingType = Union[Comment, Submission]


class BaseModNotes:
    def __init__(self, reddit: praw.Reddit) -> None: ...

    def create(
        self, *, label: Optional[str] = ..., note: str,
        redditor: Optional[RedditorType] = ...,
        subreddit: Optional[SubredditType] = ...,
        thing: Optional[Union[Comment, Submission, str]] = ...,
        **other_settings: Any) -> praw.models.ModNote: ...

    def delete(
        self, *, delete_all: bool = ..., note_id: Optional[str] = ...,
        redditor: Optional[RedditorType] = ...,
        subreddit: Optional[SubredditType] = ...) -> None: ...


class RedditModNotes(BaseModNotes):
    def __call__(
        self, *, all_notes: bool = ...,
        pairs: Optional[List[Tuple[SubredditType, RedditorType]]] = ...,
        redditors: Optional[List[RedditorType]] = ...,
        subreddits: Optional[List[SubredditType]] = ...,
        things: Optional[List[ThingType]] = ..., **generator_kwargs: Any,
        ) -> Generator['praw.models.ModNote', None, None]: ...

    def things(
        self, *things: ThingType, all_notes: Optional[bool] = ...,
        **generator_kwargs: Any,
        ) -> Generator['praw.models.ModNote', None, None]: ...


class RedditorModNotes(BaseModNotes):
    redditor: Incomplete

    def __init__(
        self, reddit: praw.Reddit, redditor: RedditorType) -> None: ...

    def subreddits(
        self, *subreddits: SubredditType, all_notes: Optional[bool] = ...,
        **generator_kwargs: Any,
        ) -> Generator['praw.models.ModNote', None, None]: ...


class SubredditModNotes(BaseModNotes):
    subreddit: Incomplete

    def __init__(
        self, reddit: praw.Reddit, subreddit: SubredditType) -> None: ...

    def redditors(
        self, *redditors: RedditorType, all_notes: Optional[bool] = ...,
        **generator_kwargs: Any,
        ) -> Generator['praw.models.ModNote', None, None]: ...
