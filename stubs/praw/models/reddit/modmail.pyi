# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,unused-argument
# pylint: disable=redefined-builtin,invalid-name
# pylint: disable=too-few-public-methods
from typing import Any, Dict, List, Optional

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...util import snake_case_keys as snake_case_keys
from .base import RedditBase as RedditBase


class ModmailConversation(RedditBase):
    STR_FIELD: str

    @classmethod
    def parse(
        cls, data: Dict[str, Any], reddit: praw.Reddit,
        convert_objects: bool = ...) -> None: ...

    id: Incomplete

    def __init__(
        self, reddit: praw.Reddit, id: Optional[str] = ...,
        mark_read: bool = ...,
        _data: Optional[Dict[str, Any]] = ...) -> None: ...

    def archive(self) -> None: ...
    def highlight(self) -> None: ...
    def mute(self, *, num_days: int = ...) -> None: ...

    def read(
        self, *, other_conversations: Optional[
            List['ModmailConversation']] = ...) -> None: ...

    def reply(
        self, *, author_hidden: bool = ..., body: str, internal: bool = ...,
        ) -> 'ModmailMessage': ...

    def unarchive(self) -> None: ...
    def unhighlight(self) -> None: ...
    def unmute(self) -> None: ...

    def unread(
        self, *,
        other_conversations: Optional[List['ModmailConversation']] = ...,
        ) -> None: ...


class ModmailObject(RedditBase):
    AUTHOR_ATTRIBUTE: str
    STR_FIELD: str
    def __setattr__(self, attribute: str, value: Any) -> None: ...


class ModmailAction(ModmailObject):
    ...


class ModmailMessage(ModmailObject):
    ...
