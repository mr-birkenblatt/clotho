from typing import Iterable

from system.msgs.message import MHash
from system.msgs.store import get_default_message_store
from system.suggest.suggest import LinkSuggester


class RandomLinkSuggester(LinkSuggester):
    def __init__(self) -> None:
        super().__init__()
        self._message_store = get_default_message_store()

    def suggest_messages(
            self,
            other: MHash,
            *,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        return self._message_store.get_random_messages(
            other, is_parent, offset, limit)
