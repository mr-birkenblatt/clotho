# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from ....const import API_PATH as API_PATH


class InboxToggleableMixin:
    def disable_inbox_replies(self) -> None: ...
    def enable_inbox_replies(self) -> None: ...
