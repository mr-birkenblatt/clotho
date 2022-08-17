# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Optional

import praw

from ...const import API_PATH as API_PATH
from .base import RedditBase as RedditBase
from .mixins import FullnameMixin as FullnameMixin
from .mixins import InboxableMixin as InboxableMixin
from .mixins import ReplyableMixin as ReplyableMixin
from .redditor import Redditor as Redditor
from .subreddit import Subreddit as Subreddit

class Message(InboxableMixin, ReplyableMixin, FullnameMixin, RedditBase):
    STR_FIELD: str
    @classmethod
    def parse(cls, data: Dict[str, Any], reddit: praw.Reddit) -> None: ...
    @property
    def parent(self) -> Optional['praw.models.Message']: ...
    @parent.setter
    def parent(self, value: Any) -> None: ...
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...
    def delete(self) -> None: ...


class SubredditMessage(Message):
    def mute(self) -> None: ...
    def unmute(self) -> None: ...
