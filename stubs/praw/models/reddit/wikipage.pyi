# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Generator, Iterator, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...util.cache import cachedproperty as cachedproperty
from ..listing.generator import ListingGenerator as ListingGenerator
from .base import RedditBase as RedditBase
from .redditor import Redditor as Redditor


class WikiPageModeration:
    wikipage: Incomplete
    def __init__(self, wikipage: 'WikiPage') -> None: ...
    def add(self, redditor: praw.models.Redditor) -> None: ...
    def remove(self, redditor: praw.models.Redditor) -> None: ...
    def revert(self) -> None: ...
    def settings(self) -> Dict[str, Any]: ...

    def update(
        self, *, listed: bool, permlevel: int,
        **other_settings: Any) -> Dict[str, Any]: ...


class WikiPage(RedditBase):
    __hash__: Incomplete
    def mod(self) -> WikiPageModeration: ...
    name: Incomplete
    subreddit: Incomplete

    def __init__(
        self, reddit: praw.Reddit, subreddit: praw.models.Subreddit, name: str,
        revision: Optional[str] = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def edit(
        self, *, content: str, reason: Optional[str] = ...,
        **other_settings: Any) -> None: ...

    def discussions(
        self, **generator_kwargs: Any,
        ) -> Iterator['praw.models.Submission']: ...

    def revision(self, revision: str) -> None: ...

    def revisions(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Generator['WikiPage', None, None]: ...
