# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, List, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...util.cache import cachedproperty as cachedproperty
from ..listing.mixins import SubredditListingMixin as SubredditListingMixin
from .base import RedditBase as RedditBase
from .redditor import Redditor as Redditor
from .subreddit import Subreddit as Subreddit
from .subreddit import SubredditStream as SubredditStream

class Multireddit(SubredditListingMixin, RedditBase):
    STR_FIELD: str
    RE_INVALID: Incomplete
    @staticmethod
    def sluggify(title: str) -> None: ...
    def stream(self) -> SubredditStream: ...
    path: Incomplete
    subreddits: Incomplete
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...
    def add(self, subreddit: praw.models.Subreddit) -> None: ...

    def copy(
        self, *,
        display_name: Optional[str] = ...) -> praw.models.Multireddit: ...

    def delete(self) -> None: ...
    def remove(self, subreddit: praw.models.Subreddit) -> None: ...

    def update(
        self, **updated_settings: Union[str, List[
            Union[str, 'praw.models.Subreddit', Dict[str, str]]]]) -> None: ...
