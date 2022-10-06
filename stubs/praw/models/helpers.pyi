# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
from typing import Any, Generator, List, Optional, Union

import praw

from ..const import API_PATH as API_PATH
from .base import PRAWBase as PRAWBase
from .reddit.draft import Draft as Draft
from .reddit.live import LiveThread as LiveThread
from .reddit.multi import Multireddit as Multireddit
from .reddit.subreddit import Subreddit as Subreddit
from .reddit.user_subreddit import UserSubreddit


class DraftHelper(PRAWBase):
    def __call__(
        self, draft_id: Optional[str] = ...) -> Union[
            List['praw.models.Draft'], 'praw.models.Draft']: ...

    def create(
        self, *, flair_id: Optional[str] = ...,
        flair_text: Optional[str] = ..., is_public_link: bool = ...,
        nsfw: bool = ..., original_content: bool = ...,
        selftext: Optional[str] = ..., send_replies: bool = ...,
        spoiler: bool = ...,
        subreddit: Optional[Union[
            str, Subreddit, UserSubreddit]] = ...,
        title: Optional[str] = ..., url: Optional[str] = ...,
        **draft_kwargs: Any,
        ) -> praw.models.Draft: ...


class LiveHelper(PRAWBase):
    def __call__(self, id: str) -> praw.models.LiveThread: ...

    def info(
        self, ids: List[str],
        ) -> Generator['praw.models.LiveThread', None, None]: ...

    def create(
        self, title: str, *, description: Optional[str] = ...,
        nsfw: bool = ..., resources: str = ...) -> praw.models.LiveThread: ...

    def now(self) -> Optional['praw.models.LiveThread']: ...


class MultiredditHelper(PRAWBase):
    def __call__(
        self, *, name: str,
        redditor: Union[str, 'praw.models.Redditor'],
        ) -> praw.models.Multireddit: ...

    def create(
        self, *, description_md: Optional[str] = ..., display_name: str,
        icon_name: Optional[str] = ..., key_color: Optional[str] = ...,
        subreddits: Union[str, Subreddit],
        visibility: str = ..., weighting_scheme: str = ...,
        ) -> praw.models.Multireddit: ...


class SubredditHelper(PRAWBase):
    def __call__(self, display_name: str) -> Subreddit: ...

    def create(
        self, name: str, *, link_type: str = ..., subreddit_type: str = ...,
        title: Optional[str] = ..., wikimode: str = ...,
        **other_settings: Optional[str]) -> Subreddit: ...
