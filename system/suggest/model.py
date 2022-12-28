from typing import Iterable

from system.embedding.store import get_embed_store
from system.msgs.message import MHash
from system.namespace.namespace import Namespace
from system.suggest.suggest import LinkSuggester


class ModelLinkSuggester(LinkSuggester):
    def __init__(self, namespace: Namespace, count: int) -> None:
        super().__init__(namespace)
        self._embed = get_embed_store(namespace)
        self._count = count

    def suggest_messages(
            self,
            other: MHash,
            *,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        raise NotImplementedError()
