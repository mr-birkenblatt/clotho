# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Dict, Iterator, List, Optional, Union

import praw

from ..const import API_PATH as API_PATH
from ..exceptions import ReadOnlyException as ReadOnlyException
from ..models import Preferences as Preferences
from ..util.cache import cachedproperty as cachedproperty
from .base import PRAWBase as PRAWBase
from .listing.generator import ListingGenerator as ListingGenerator
from .reddit.redditor import Redditor as Redditor
from .reddit.subreddit import Subreddit as Subreddit


class User(PRAWBase):
    def preferences(self) -> praw.models.Preferences: ...
    def __init__(self, reddit: praw.Reddit) -> None: ...
    def blocked(self) -> List['praw.models.Redditor']: ...

    def contributor_subreddits(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def friends(
        self, *, user: Optional[Union[str, 'praw.models.Redditor']] = ...,
        ) -> Union[List['praw.models.Redditor'], 'praw.models.Redditor']: ...

    def karma(self) -> Dict['praw.models.Subreddit', Dict[str, int]]: ...

    def me(
        self, *,
        use_cache: bool = ...) -> Optional['praw.models.Redditor']: ...

    def moderator_subreddits(
        self,
        **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def multireddits(self) -> List['praw.models.Multireddit']: ...

    def pin(
        self, submission: praw.models.Submission, *,
        num: int = ..., state: bool = ...) -> None: ...

    def subreddits(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def trusted(self) -> List['praw.models.Redditor']: ...
