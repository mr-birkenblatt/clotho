# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods
from typing import Optional

import praw
from _typeshed import Incomplete

from .editable import EditableMixin as EditableMixin
from .fullname import FullnameMixin as FullnameMixin
from .gildable import GildableMixin as GildableMixin
from .inboxable import InboxableMixin as InboxableMixin
from .inboxtoggleable import InboxToggleableMixin as InboxToggleableMixin
from .messageable import MessageableMixin as MessageableMixin
from .modnote import ModNoteMixin as ModNoteMixin
from .replyable import ReplyableMixin as ReplyableMixin
from .reportable import ReportableMixin as ReportableMixin
from .savable import SavableMixin as SavableMixin
from .votable import VotableMixin as VotableMixin


class ThingModerationMixin(ModNoteMixin):
    REMOVAL_MESSAGE_API: Incomplete

    def approve(self) -> None: ...
    def distinguish(self, *, how: str = ..., sticky: bool = ...) -> None: ...
    def ignore_reports(self) -> None: ...
    def lock(self) -> None: ...

    def remove(
        self, *, mod_note: str = ..., spam: bool = ...,
        reason_id: Optional[str] = ...) -> None: ...

    def send_removal_message(
        self, *, message: str, title: str = ..., type: str = ...,
        ) -> Optional['praw.models.Comment']: ...

    def undistinguish(self) -> None: ...
    def unignore_reports(self) -> None: ...
    def unlock(self) -> None: ...


class UserContentMixin(
        EditableMixin, GildableMixin, InboxToggleableMixin, ReplyableMixin,
        ReportableMixin, SavableMixin, VotableMixin):
    ...
