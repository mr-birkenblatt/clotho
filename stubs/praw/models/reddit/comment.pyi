from typing import Any, Dict, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import ClientException as ClientException
from ...exceptions import InvalidURL as InvalidURL
from ...util.cache import cachedproperty as cachedproperty
from ..comment_forest import CommentForest as CommentForest
from .base import RedditBase as RedditBase
from .mixins import FullnameMixin as FullnameMixin
from .mixins import InboxableMixin as InboxableMixin
from .mixins import ThingModerationMixin as ThingModerationMixin
from .mixins import UserContentMixin as UserContentMixin
from .redditor import Redditor as Redditor

class Comment(InboxableMixin, UserContentMixin, FullnameMixin, RedditBase):
    MISSING_COMMENT_MESSAGE: str
    STR_FIELD: str
    @staticmethod
    def id_from_url(url: str) -> str: ...
    @property
    def is_root(self) -> bool: ...
    def mod(self) -> praw.models.reddit.comment.CommentModeration: ...
    @property
    def replies(self) -> CommentForest: ...
    @property
    def submission(self) -> praw.models.Submission: ...
    @submission.setter
    def submission(self, submission: praw.models.Submission) -> None: ...
    id: Incomplete
    def __init__(self, reddit: praw.Reddit, id: Optional[str] = ..., url: Optional[str] = ..., _data: Optional[Dict[str, Any]] = ...) -> None: ...
    def __setattr__(self, attribute: str, value: Union[str, Redditor, CommentForest, 'praw.models.Subreddit']) -> None: ...
    def parent(self) -> Union['Comment', 'praw.models.Submission']: ...
    def refresh(self) -> None: ...

class CommentModeration(ThingModerationMixin):
    REMOVAL_MESSAGE_API: str
    thing: Incomplete
    def __init__(self, comment: praw.models.Comment) -> None: ...
    def show(self) -> None: ...
