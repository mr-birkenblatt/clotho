# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods
from typing import Any, Dict, List, Optional, Union

import praw
from _typeshed import Incomplete
from praw.models.reddit.submission import Award

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

    id: str
    body: str
    body_html: str
    parent_id: str

    permalink: str
    name: str

    score: int
    author: Optional[Redditor]
    author_fullname: str
    total_awards_received: int
    subreddit_name_prefixed: str
    num_reports: int
    fullname: str
    downs: int
    ups: int

    created_utc: float
    all_awardings: List[Award]
    depth: int

    is_submitter: bool

    def __init__(
        self, reddit: praw.Reddit, id: Optional[str] = ...,
        url: Optional[str] = ..., _data: Optional[Dict[str, Any]] = ...,
        ) -> None: ...

    def __setattr__(
        self, attribute: str, value: Union[
            str, Redditor, CommentForest, 'praw.models.Subreddit'],
        ) -> None: ...

    def parent(self) -> Union['Comment', 'praw.models.Submission']: ...

    def refresh(self) -> None: ...


class CommentModeration(ThingModerationMixin):
    REMOVAL_MESSAGE_API: str
    thing: Incomplete
    def __init__(self, comment: praw.models.Comment) -> None: ...
    def show(self) -> None: ...
