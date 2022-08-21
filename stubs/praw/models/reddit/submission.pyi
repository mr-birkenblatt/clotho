# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
from typing import Any, Dict, List, Optional, TypedDict, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import InvalidURL as InvalidURL
from ...util import cachedproperty as cachedproperty
from ..comment_forest import CommentForest as CommentForest
from ..listing.listing import Listing as Listing
from ..listing.mixins import SubmissionListingMixin as SubmissionListingMixin
from .base import RedditBase as RedditBase
from .mixins import FullnameMixin as FullnameMixin
from .mixins import ModNoteMixin as ModNoteMixin
from .mixins import ThingModerationMixin as ThingModerationMixin
from .mixins import UserContentMixin as UserContentMixin
from .poll import PollData as PollData
from .redditor import Redditor as Redditor
from .subreddit import Subreddit as Subreddit


Icon = TypedDict('Icon', {
    "url": str,
    "width": int,
    "height": int,
})

Award = TypedDict('Award', {
    "giver_coin_reward": None,
    "subreddit_id": None,
    "is_new": bool,
    "days_of_drip_extension": None,
    "coin_price": int,
    "id": str,
    "penny_donate": None,
    "coint_reward": int,
    "icon_url": str,
    "days_of_premious": None,
    "icon_height": int,
    "tiers_by_required_awardings": None,
    "resized_icons": List[Icon],
    "icon_width": int,
    "static_icon_width": int,
    "start_date": None,
    "is_enabled": bool,
    "awardings_required_to_grant_benefits": None,
    "description": str,
    "end_date": None,
    "sticky_duration_seconds": None,
    "subreddit_coin_reward": int,
    "count": int,
    "static_icon_height": int,
    "name": str,
    "resized_static_icons": List[Icon],
    "icon_format": None,
    "award_sub_type": str,
    "penny_price": None,
    "award_type": str,
    "static_icon_url": str,
})


class SubmissionFlair:
    submission: Incomplete
    def __init__(self, submission: praw.models.Submission) -> None: ...
    def choices(self) -> List[Dict[str, Union[bool, list, str]]]: ...

    def select(
        self, flair_template_id: str, *,
        text: Optional[str] = ...) -> None: ...


class SubmissionModeration(ThingModerationMixin, ModNoteMixin):
    REMOVAL_MESSAGE_API: str
    thing: Incomplete
    def __init__(self, submission: praw.models.Submission) -> None: ...
    def contest_mode(self, *, state: bool = ...) -> None: ...

    def flair(
        self, *, css_class: str = ...,
        flair_template_id: Optional[str] = ...,
        text: str = ...) -> None: ...

    def nsfw(self) -> None: ...
    def set_original_content(self) -> None: ...
    def sfw(self) -> None: ...
    def spoiler(self) -> None: ...
    def sticky(self, *, bottom: bool = ..., state: bool = ...) -> None: ...
    def suggested_sort(self, *, sort: str = ...) -> None: ...
    def unset_original_content(self) -> None: ...
    def unspoiler(self) -> None: ...
    def update_crowd_control_level(self, level: int) -> None: ...


class Submission(
        SubmissionListingMixin, UserContentMixin, FullnameMixin, RedditBase):
    STR_FIELD: str

    @staticmethod
    def id_from_url(url: str) -> str: ...

    @property
    def comments(self) -> CommentForest: ...
    def flair(self) -> SubmissionFlair: ...
    def mod(self) -> SubmissionModeration: ...

    @property
    def shortlink(self) -> str: ...
    comment_limit: int
    comment_sort: str

    permalink: str
    name: str
    id: str
    url: str
    title: str
    score: int
    upvote_ratio: float
    author: Optional[Redditor]
    author_fullname: str
    total_awards_received: int
    subreddit_name_prefixed: str
    subreddit: praw.models.Subreddit
    selftext_html: str
    selftext: str
    num_comments: int
    num_crossposts: int
    num_duplicates: int
    num_reports: int
    fullname: str
    downs: int
    ups: int

    created_utc: float
    all_awardings: List[Award]

    is_created_from_ads_ui: bool
    is_crosspostable: bool
    is_meta: bool
    is_original_content: bool
    is_reddit_media_domain: bool
    is_robot_indexable: bool
    is_self: bool
    is_video: bool

    def __init__(
        self, reddit: praw.Reddit, id: Optional[str] = ...,
        url: Optional[str] = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def __setattr__(self, attribute: str, value: Any) -> None: ...
    def mark_visited(self) -> None: ...

    def hide(
        self, *, other_submissions: Optional[
            List['praw.models.Submission']] = ...) -> None: ...

    def unhide(
        self, *, other_submissions: Optional[
            List['praw.models.Submission']] = ...) -> None: ...

    def crosspost(
        self, subreddit: praw.models.Subreddit, *,
        flair_id: Optional[str] = ..., flair_text: Optional[str] = ...,
        nsfw: bool = ..., send_replies: bool = ..., spoiler: bool = ...,
        title: Optional[str] = ...) -> praw.models.Submission: ...
