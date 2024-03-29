# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods
from typing import Any, Dict, Iterator, List, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...util.cache import cachedproperty as cachedproperty
from ..list.redditor import RedditorList as RedditorList
from ..listing.generator import ListingGenerator as ListingGenerator
from ..util import stream_generator as stream_generator
from .base import RedditBase as RedditBase
from .mixins import FullnameMixin as FullnameMixin
from .redditor import Redditor as Redditor


class LiveContributorRelationship:
    def __call__(self) -> List['praw.models.Redditor']: ...
    thread: Incomplete
    def __init__(self, thread: praw.models.LiveThread) -> None: ...
    def accept_invite(self) -> None: ...

    def invite(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...) -> None: ...

    def leave(self) -> None: ...

    def remove(
        self, redditor: Union[str, 'praw.models.Redditor']) -> None: ...

    def remove_invite(
        self, redditor: Union[str, 'praw.models.Redditor']) -> None: ...

    def update(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...) -> None: ...

    def update_invite(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...) -> None: ...


class LiveThread(RedditBase):
    STR_FIELD: str
    def contrib(self) -> praw.models.reddit.live.LiveThreadContribution: ...

    def contributor(
        self) -> praw.models.reddit.live.LiveContributorRelationship: ...

    def stream(self) -> praw.models.reddit.live.LiveThreadStream: ...

    def __eq__(
        self, other: Union[str, 'praw.models.LiveThread', object]) -> bool: ...

    def __getitem__(self, update_id: str) -> praw.models.LiveUpdate: ...
    def __hash__(self) -> int: ...

    id: Incomplete

    def __init__(
        self, reddit: praw.Reddit, id: Optional[str] = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def discussions(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Submission']: ...

    def report(self, type: str) -> None: ...

    def updates(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.LiveUpdate']: ...


class LiveThreadContribution:
    thread: Incomplete
    def __init__(self, thread: praw.models.LiveThread) -> None: ...
    def add(self, body: str) -> None: ...
    def close(self) -> None: ...

    def update(
        self, *, description: Optional[str] = ..., nsfw: Optional[bool] = ...,
        resources: Optional[str] = ..., title: Optional[str] = ...,
        **other_settings: Optional[str]) -> None: ...


class LiveThreadStream:
    live_thread: Incomplete
    def __init__(self, live_thread: praw.models.LiveThread) -> None: ...

    def updates(
        self, **stream_options: Dict[str, Any],
        ) -> Iterator['praw.models.LiveUpdate']: ...


class LiveUpdate(FullnameMixin, RedditBase):
    STR_FIELD: str
    def contrib(self) -> praw.models.reddit.live.LiveUpdateContribution: ...
    @property
    def thread(self) -> LiveThread: ...

    id: Incomplete

    def __init__(
        self, reddit: praw.Reddit, thread_id: Optional[str] = ...,
        update_id: Optional[str] = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def __setattr__(self, attribute: str, value: Any) -> None: ...


class LiveUpdateContribution:
    update: Incomplete
    def __init__(self, update: praw.models.LiveUpdate) -> None: ...
    def remove(self) -> None: ...
    def strike(self) -> None: ...
