
from typing import Iterable, Literal, TypedDict

from system.links.link import Link
from system.links.store import get_link_store, LinkStore
from system.msgs.message import MHash
from system.namespace.module import ModuleBase
from system.namespace.namespace import Namespace


class LinkSuggester(ModuleBase):
    def __init__(self, namespace: Namespace) -> None:
        super().__init__()
        self._link_store = get_link_store(namespace)

    @staticmethod
    def module_name() -> str:
        return "suggest"

    def get_link_store(self) -> LinkStore:
        return self._link_store

    def suggest_messages(
            self,
            other: MHash,
            *,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        raise NotImplementedError()

    def suggest_links(
            self,
            other: MHash,
            *,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[Link]:
        if limit <= 0:
            return []
        link_store = self.get_link_store()
        return [
            link_store.get_link(
                other if is_parent else cur,
                cur if is_parent else other)
            for cur in self.suggest_messages(
                other, is_parent=is_parent, offset=offset, limit=limit)
        ]

    def max_suggestions(self) -> int | None:
        raise NotImplementedError()

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()


SUGGESTER_STORE: dict[Namespace, list[LinkSuggester]] = {}


def get_link_suggesters(namespace: Namespace) -> list[LinkSuggester]:
    res = SUGGESTER_STORE.get(namespace)
    if res is None:
        res = list(create_link_suggesters(namespace))
        SUGGESTER_STORE[namespace] = res
    return res


RandomSuggestModule = TypedDict('RandomSuggestModule', {
    "name": Literal["random"],
})
ModelSuggestModule = TypedDict('ModelSuggestModule', {
    "name": Literal["model"],
    "count": int,
})
SuggestModule = RandomSuggestModule | ModelSuggestModule


def create_link_suggesters(namespace: Namespace) -> Iterable[LinkSuggester]:
    for sobj in namespace.get_suggest_module():
        if sobj["name"] == "random":
            from system.suggest.random import RandomLinkSuggester
            yield RandomLinkSuggester(namespace)
        elif sobj["name"] == "model":
            from system.suggest.model import ModelLinkSuggester
            yield ModelLinkSuggester(namespace, sobj["count"])
        else:
            raise ValueError(f"unknown link suggester: {sobj}")
