from typing import Any, Dict, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import ClientException as ClientException
from .base import RedditBase as RedditBase
from .subreddit import Subreddit as Subreddit
from .user_subreddit import UserSubreddit as UserSubreddit

class Draft(RedditBase):
    STR_FIELD: str
    id: Incomplete
    def __init__(self, reddit: praw.Reddit, id: Optional[str] = ..., _data: Dict[str, Any] = ...) -> None: ...
    def delete(self) -> None: ...
    def update(self, *, flair_id: Optional[str] = ..., flair_text: Optional[str] = ..., is_public_link: Optional[bool] = ..., nsfw: Optional[bool] = ..., original_content: Optional[bool] = ..., selftext: Optional[str] = ..., send_replies: Optional[bool] = ..., spoiler: Optional[bool] = ..., subreddit: Optional[Union[str, 'praw.models.Subreddit', 'praw.models.UserSubreddit']] = ..., title: Optional[str] = ..., url: Optional[str] = ..., **draft_kwargs) -> None: ...
    def submit(self, *, flair_id: Optional[str] = ..., flair_text: Optional[str] = ..., nsfw: Optional[bool] = ..., selftext: Optional[str] = ..., spoiler: Optional[bool] = ..., subreddit: Optional[Union[str, 'praw.models.Subreddit', 'praw.models.UserSubreddit']] = ..., title: Optional[str] = ..., url: Optional[str] = ..., **submit_kwargs) -> praw.models.Submission: ...
