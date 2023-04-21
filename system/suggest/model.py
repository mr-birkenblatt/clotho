from typing import Iterable

from system.embedding.store import get_embed_store
from system.msgs.message import MHash
from system.msgs.store import get_message_store
from system.namespace.namespace import Namespace
from system.suggest.suggest import LinkSuggester


class ModelLinkSuggester(LinkSuggester):
    def __init__(self, namespace: Namespace, count: int) -> None:
        super().__init__(namespace)
        self._message_store = get_message_store(namespace)
        self._embed = get_embed_store(namespace)
        self._count = count

    def suggest_messages(
            self,
            other: MHash,
            *,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        embed = self._embed
        precise = False
        no_cache = False
        other_embed = embed.get_embedding(
            self._message_store,
            "parent" if is_parent else "child",
            other,
            no_index=precise,
            no_cache=no_cache)
        res = list(embed.get_closest(
            "child" if is_parent else "parent",
            other_embed,
            min(self._count, offset + limit),
            precise=precise,
            no_cache=no_cache))
        return res[offset:]

    def max_suggestions(self) -> int | None:
        return self._count

    @staticmethod
    def get_name() -> str:
        return "model"
