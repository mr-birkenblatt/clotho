from typing import Dict, Union

import praw

from ...util.cache import cachedproperty as cachedproperty
from .subreddit import Subreddit as Subreddit
from .subreddit import SubredditModeration as SubredditModeration

class UserSubreddit(Subreddit):
    def __init__(self, reddit: praw.Reddit, *args, **kwargs): ...
    def mod(self) -> praw.models.reddit.user_subreddit.UserSubredditModeration: ...
    def __getitem__(self, item): ...

class UserSubredditModeration(SubredditModeration):
    def update(self, **settings: Union[str, int, bool]) -> Dict[str, Union[str, int, bool]]: ...
