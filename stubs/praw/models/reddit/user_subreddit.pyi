# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
# pylint: disable=too-few-public-methods
from typing import Any, Dict, Union

import praw

from ...util.cache import cachedproperty as cachedproperty
from .subreddit import Subreddit as Subreddit
from .subreddit import SubredditModeration as SubredditModeration


class UserSubreddit(Subreddit):
    def __init__(
        self, reddit: praw.Reddit, *args: Any, **kwargs: Any) -> None: ...

    def mod(
        self) -> praw.models.reddit.user_subreddit.UserSubredditModeration: ...

    def __getitem__(self, item: Any) -> None: ...


class UserSubredditModeration(SubredditModeration):
    def update(
        self, **settings: Union[str, int, bool],
        ) -> Dict[str, Union[str, int, bool]]: ...
