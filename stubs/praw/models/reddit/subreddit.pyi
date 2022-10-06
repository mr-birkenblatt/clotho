# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
# pylint: disable=arguments-differ
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...const import JPEG_HEADER as JPEG_HEADER
from ...exceptions import ClientException as ClientException
from ...exceptions import InvalidFlairTemplateID as InvalidFlairTemplateID
from ...exceptions import MediaPostFailed as MediaPostFailed
from ...exceptions import RedditAPIException as RedditAPIException
from ...exceptions import TooLargeMediaException as TooLargeMediaException
from ...exceptions import WebSocketException as WebSocketException
from ...util import cachedproperty as cachedproperty
from ..listing.generator import ListingGenerator as ListingGenerator
from ..listing.mixins import SubredditListingMixin as SubredditListingMixin
from ..util import permissions_string as permissions_string
from ..util import stream_generator as stream_generator
from .base import RedditBase as RedditBase
from .emoji import SubredditEmoji as SubredditEmoji
from .mixins import FullnameMixin as FullnameMixin
from .mixins import MessageableMixin as MessageableMixin
from .modmail import ModmailConversation as ModmailConversation
from .removal_reasons import SubredditRemovalReasons as SubredditRemovalReasons
from .rules import SubredditRules as SubredditRules
from .widgets import SubredditWidgets as SubredditWidgets
from .widgets import WidgetEncoder as WidgetEncoder
from .wikipage import WikiPage as WikiPage


class Subreddit(
        MessageableMixin, SubredditListingMixin, FullnameMixin, RedditBase):
    STR_FIELD: str
    MESSAGE_PREFIX: str

    def banned(self) -> praw.models.reddit.subreddit.SubredditRelationship: ...

    def collections(
        self) -> praw.models.reddit.collections.SubredditCollections: ...

    def contributor(
        self) -> praw.models.reddit.subreddit.ContributorRelationship: ...

    def emoji(self) -> SubredditEmoji: ...

    def filters(self) -> praw.models.reddit.subreddit.SubredditFilters: ...

    def flair(self) -> praw.models.reddit.subreddit.SubredditFlair: ...

    def mod(self) -> 'SubredditModeration': ...

    def moderator(
        self) -> praw.models.reddit.subreddit.ModeratorRelationship: ...

    def modmail(self) -> praw.models.reddit.subreddit.Modmail: ...

    def muted(self) -> praw.models.reddit.subreddit.SubredditRelationship: ...

    def quaran(self) -> praw.models.reddit.subreddit.SubredditQuarantine: ...

    def rules(self) -> 'SubredditRules': ...

    def stream(self) -> praw.models.reddit.subreddit.SubredditStream: ...

    def stylesheet(
        self) -> praw.models.reddit.subreddit.SubredditStylesheet: ...

    def widgets(self) -> praw.models.SubredditWidgets: ...

    def wiki(self) -> praw.models.reddit.subreddit.SubredditWiki: ...

    display_name: str
    id: str
    name: str
    fullname: str

    def __init__(
        self, reddit: praw.Reddit, display_name: Optional[str] = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def post_requirements(self) -> Dict[str, Union[str, int, bool]]: ...

    def random(self) -> Union['praw.models.Submission', None]: ...

    def search(
        self, query: str, *, sort: str = ..., syntax: str = ...,
        time_filter: str = ..., **generator_kwargs: Any,
        ) -> Iterator['praw.models.Submission']: ...

    def sticky(self, *, number: int = ...) -> praw.models.Submission: ...

    def submit(
        self, title: str, *, collection_id: Optional[str] = ...,
        discussion_type: Optional[str] = ..., draft_id: Optional[str] = ...,
        flair_id: Optional[str] = ..., flair_text: Optional[str] = ...,
        inline_media: Optional[Dict[str, 'praw.models.InlineMedia']] = ...,
        nsfw: bool = ..., resubmit: bool = ..., selftext: Optional[str] = ...,
        send_replies: bool = ..., spoiler: bool = ...,
        url: Optional[str] = ...) -> praw.models.Submission: ...

    def submit_gallery(
        self, title: str, images: List[Dict[str, str]], *,
        collection_id: Optional[str] = ...,
        discussion_type: Optional[str] = ..., flair_id: Optional[str] = ...,
        flair_text: Optional[str] = ..., nsfw: bool = ...,
        send_replies: bool = ..., spoiler: bool = ...) -> None: ...

    def submit_image(
        self, title: str, image_path: str, *,
        collection_id: Optional[str] = ...,
        discussion_type: Optional[str] = ..., flair_id: Optional[str] = ...,
        flair_text: Optional[str] = ..., nsfw: bool = ...,
        resubmit: bool = ..., send_replies: bool = ..., spoiler: bool = ...,
        timeout: int = ..., without_websockets: bool = ...) -> None: ...

    def submit_poll(
        self, title: str, *, collection_id: Optional[str] = ...,
        discussion_type: Optional[str] = ..., duration: int,
        flair_id: Optional[str] = ..., flair_text: Optional[str] = ...,
        nsfw: bool = ..., options: List[str], resubmit: bool = ...,
        selftext: str, send_replies: bool = ...,
        spoiler: bool = ...) -> None: ...

    def submit_video(
        self, title: str, video_path: str, *,
        collection_id: Optional[str] = ...,
        discussion_type: Optional[str] = ..., flair_id: Optional[str] = ...,
        flair_text: Optional[str] = ..., nsfw: bool = ...,
        resubmit: bool = ..., send_replies: bool = ..., spoiler: bool = ...,
        thumbnail_path: Optional[str] = ..., timeout: int = ...,
        videogif: bool = ..., without_websockets: bool = ...) -> None: ...

    def subscribe(
        self, *,
        other_subreddits: Optional[List['praw.models.Subreddit']] = ...,
        ) -> None: ...

    def traffic(self) -> Dict[str, List[List[int]]]: ...

    def unsubscribe(
        self, *,
        other_subreddits: Optional[List['praw.models.Subreddit']] = ...,
        ) -> None: ...


class SubredditFilters:
    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def __iter__(self) -> Generator['praw.models.Subreddit', None, None]: ...

    def add(self, subreddit: Union['praw.models.Subreddit', str]) -> None: ...

    def remove(
        self, subreddit: Union['praw.models.Subreddit', str]) -> None: ...


class SubredditFlair:
    def link_templates(
        self) -> praw.models.reddit.subreddit.SubredditLinkFlairTemplates: ...

    def templates(
        self,
        ) -> praw.models.reddit.subreddit.SubredditRedditorFlairTemplates: ...

    def __call__(
        self, redditor: Optional[Union['praw.models.Redditor', str]] = ...,
        **generator_kwargs: Any) -> Iterator['praw.models.Redditor']: ...

    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def configure(
        self, *, link_position: str = ..., link_self_assign: bool = ...,
        position: str = ..., self_assign: bool = ...,
        **settings: Any) -> None: ...

    def delete(self, redditor: Union['praw.models.Redditor', str]) -> None: ...

    def delete_all(
        self) -> List[Dict[str, Union[str, bool, Dict[str, str]]]]: ...

    def set(
        self, redditor: Union['praw.models.Redditor', str], *,
        css_class: str = ..., flair_template_id: Optional[str] = ...,
        text: str = ...) -> None: ...

    def update(
        self, flair_list: Iterator[Union[
            str, 'praw.models.Redditor',
            Dict[str, Union[str, 'praw.models.Redditor']]]],
        *, text: str = ..., css_class: str = ...,
        ) -> List[Dict[str, Union[str, bool, Dict[str, str]]]]: ...


class SubredditFlairTemplates:
    @staticmethod
    def flair_type(is_link: bool) -> str: ...

    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def __iter__(self) -> None: ...

    def delete(self, template_id: str) -> None: ...

    def update(
        self, template_id: str, *, allowable_content: Optional[str] = ...,
        background_color: Optional[str] = ..., css_class: Optional[str] = ...,
        fetch: bool = ..., max_emojis: Optional[int] = ...,
        mod_only: Optional[bool] = ..., text: Optional[str] = ...,
        text_color: Optional[str] = ...,
        text_editable: Optional[bool] = ...) -> None: ...


class SubredditRedditorFlairTemplates(SubredditFlairTemplates):
    def add(
        self, text: str, *, allowable_content: Optional[str] = ...,
        background_color: Optional[str] = ..., css_class: str = ...,
        max_emojis: Optional[int] = ..., mod_only: Optional[bool] = ...,
        text_color: Optional[str] = ...,
        text_editable: bool = ...) -> None: ...

    def clear(self) -> None: ...


class SubredditLinkFlairTemplates(SubredditFlairTemplates):
    def add(
        self, text: str, *, allowable_content: Optional[str] = ...,
        background_color: Optional[str] = ..., css_class: str = ...,
        max_emojis: Optional[int] = ..., mod_only: Optional[bool] = ...,
        text_color: Optional[str] = ...,
        text_editable: bool = ...) -> None: ...

    def clear(self) -> None: ...

    def user_selectable(
        self) -> Generator[Dict[str, Union[str, bool]], None, None]: ...


class SubredditModeration:
    def notes(self) -> praw.models.SubredditModNotes: ...

    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...
    def accept_invite(self) -> None: ...

    def edited(
        self, *, only: Optional[str] = ..., **generator_kwargs: Any,
        ) -> Iterator[
            Union['praw.models.Comment', 'praw.models.Submission']]: ...

    def inbox(
        self, **generator_kwargs: Any,
        ) -> Iterator['praw.models.SubredditMessage']: ...

    def log(
        self, *, action: Optional[str] = ...,
        mod: Optional[Union['praw.models.Redditor', str]] = ...,
        **generator_kwargs: Any) -> Iterator['praw.models.ModAction']: ...

    def modqueue(
        self, *, only: Optional[str] = ..., **generator_kwargs: Any,
        ) -> Iterator[Union[
            'praw.models.Submission', 'praw.models.Comment']]: ...

    def stream(
        self) -> praw.models.reddit.subreddit.SubredditModerationStream: ...

    def removal_reasons(self) -> SubredditRemovalReasons: ...

    def reports(
        self, *, only: Optional[str] = ..., **generator_kwargs: Any,
        ) -> Iterator[Union[
            'praw.models.Submission', 'praw.models.Comment']]: ...

    def settings(self) -> Dict[str, Union[str, int, bool]]: ...

    def spam(
        self, *, only: Optional[str] = ..., **generator_kwargs: Any,
        ) -> Iterator[Union[
            'praw.models.Submission', 'praw.models.Comment']]: ...

    def unmoderated(
        self, **generator_kwargs: Any,
        ) -> Iterator['praw.models.Submission']: ...

    def unread(
        self, **generator_kwargs: Any,
        ) -> Iterator['praw.models.SubredditMessage']: ...

    def update(
        self, **settings: Union[str, int, bool],
        ) -> Dict[str, Union[str, int, bool]]: ...


class SubredditModerationStream:
    subreddit: Incomplete
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def edited(
        self, *, only: Optional[str] = ..., **stream_options: Any,
        ) -> Generator[Union[
            'praw.models.Comment', 'praw.models.Submission'], None, None]: ...

    def log(
        self, *, action: Optional[str] = ...,
        mod: Optional[Union[str, 'praw.models.Redditor']] = ...,
        **stream_options: Any) -> Generator[
            'praw.models.ModAction', None, None]: ...

    def modmail_conversations(
        self, *, other_subreddits: Optional[
            List['praw.models.Subreddit']] = ...,
        sort: Optional[str] = ..., state: Optional[str] = ...,
        **stream_options: Any,
        ) -> Generator[ModmailConversation, None, None]: ...

    def modqueue(
        self, *, only: Optional[str] = ..., **stream_options: Any,
        ) -> Generator[Union[
            'praw.models.Comment', 'praw.models.Submission'], None, None]: ...

    def reports(
        self, *, only: Optional[str] = ..., **stream_options: Any,
        ) -> Generator[Union[
            'praw.models.Comment', 'praw.models.Submission'], None, None]: ...

    def spam(
        self, *, only: Optional[str] = ..., **stream_options: Any,
        ) -> Generator[Union[
            'praw.models.Comment', 'praw.models.Submission'], None, None]: ...

    def unmoderated(
        self, **stream_options: Any) -> Generator[
            'praw.models.Submission', None, None]: ...

    def unread(
        self, **stream_options: Any) -> Generator[
            'praw.models.SubredditMessage', None, None]: ...


class SubredditQuarantine:
    subreddit: Incomplete
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...
    def opt_in(self) -> None: ...
    def opt_out(self) -> None: ...


class SubredditRelationship:
    def __call__(
        self, redditor: Optional[Union[str, 'praw.models.Redditor']] = ...,
        **generator_kwargs: Any) -> Iterator['praw.models.Redditor']: ...

    relationship: Incomplete
    subreddit: Incomplete

    def __init__(
        self, subreddit: praw.models.Subreddit, relationship: str) -> None: ...

    def add(
        self, redditor: Union[str, 'praw.models.Redditor'],
        **other_settings: Any) -> None: ...

    def remove(self, redditor: Union[str, 'praw.models.Redditor']) -> None: ...


class ContributorRelationship(SubredditRelationship):
    def leave(self) -> None: ...


class ModeratorRelationship(SubredditRelationship):
    PERMISSIONS: Incomplete

    def add(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...,
        **other_settings: Any) -> None: ...

    def invite(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...,
        **other_settings: Any) -> None: ...

    def invited(
        self, *, redditor: Optional[Union[str, 'praw.models.Redditor']] = ...,
        **generator_kwargs: Any) -> Iterator['praw.models.Redditor']: ...

    def leave(self) -> None: ...

    def remove_invite(
        self, redditor: Union[str, 'praw.models.Redditor']) -> None: ...

    def update(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...) -> None: ...

    def update_invite(
        self, redditor: Union[str, 'praw.models.Redditor'], *,
        permissions: Optional[List[str]] = ...) -> None: ...


class Modmail:
    def __call__(
        self, id: Optional[str] = ..., mark_read: bool = ...) -> None: ...

    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def bulk_read(
        self, *, other_subreddits: Optional[List[Union[
            'praw.models.Subreddit', str]]] = ...,
        state: Optional[str] = ...) -> List[ModmailConversation]: ...

    def conversations(
        self, *, after: Optional[str] = ...,
        other_subreddits: Optional[List['praw.models.Subreddit']] = ...,
        sort: Optional[str] = ..., state: Optional[str] = ...,
        **generator_kwargs: Any) -> Iterator[ModmailConversation]: ...

    def create(
        self, *, author_hidden: bool = ..., body: str,
        recipient: Union[str, 'praw.models.Redditor'],
        subject: str) -> ModmailConversation: ...

    def subreddits(self) -> Generator['praw.models.Subreddit', None, None]: ...
    def unread_count(self) -> Dict[str, int]: ...


class SubredditStream:
    subreddit: Incomplete
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def comments(
        self, **stream_options: Any,
        ) -> Generator['praw.models.Comment', None, None]: ...

    def submissions(
        self, **stream_options: Any,
        ) -> Generator['praw.models.Submission', None, None]: ...


class SubredditStylesheet:
    def __call__(self) -> praw.models.Stylesheet: ...
    subreddit: Incomplete
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...
    def delete_banner(self) -> None: ...
    def delete_banner_additional_image(self) -> None: ...
    def delete_banner_hover_image(self) -> None: ...
    def delete_header(self) -> None: ...
    def delete_image(self, name: str) -> None: ...
    def delete_mobile_header(self) -> None: ...
    def delete_mobile_icon(self) -> None: ...
    def update(
        self, stylesheet: str, *, reason: Optional[str] = ...) -> None: ...

    def upload(self, *, image_path: str, name: str) -> Dict[str, str]: ...
    def upload_banner(self, image_path: str) -> None: ...
    def upload_banner_additional_image(
        self, image_path: str, *, align: Optional[str] = ...) -> None: ...

    def upload_banner_hover_image(self, image_path: str) -> None: ...
    def upload_header(self, image_path: str) -> Dict[str, str]: ...
    def upload_mobile_header(self, image_path: str) -> Dict[str, str]: ...
    def upload_mobile_icon(self, image_path: str) -> Dict[str, str]: ...


class SubredditWiki:
    def __getitem__(self, page_name: str) -> WikiPage: ...

    banned: Incomplete
    contributor: Incomplete
    subreddit: Incomplete

    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...

    def __iter__(self) -> Generator[WikiPage, None, None]: ...

    def create(
        self, *, content: str, name: str, reason: Optional[str] = ...,
        **other_settings: Any) -> None: ...

    def revisions(
        self, **generator_kwargs: Any) -> Generator[Dict[str, Optional[
            Union['praw.models.Redditor', WikiPage, str, int, bool]]],
            None, None]: ...
