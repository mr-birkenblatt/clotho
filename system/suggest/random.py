from typing import Iterable

from system.msgs.message import MHash
from system.msgs.store import get_message_store
from system.namespace.namespace import Namespace
from system.suggest.suggest import LinkSuggester


class RandomLinkSuggester(LinkSuggester):
    def __init__(self, namespace: Namespace) -> None:
        super().__init__(namespace)
        self._message_store = get_message_store(namespace)

    def suggest_messages(
            self,
            other: MHash,
            *,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        return self._message_store.get_random_messages(
            other, is_parent, offset, limit)

    def max_suggestions(self) -> int | None:
        return None

    @staticmethod
    def get_name() -> str:
        return "random"
