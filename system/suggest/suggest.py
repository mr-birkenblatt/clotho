
from typing import Iterable, Literal, TypedDict

from system.links.link import Link
from system.links.store import get_link_store, LinkStore
from system.msgs.message import MHash
from system.namespace.namespace import Namespace


class LinkSuggester:
    def __init__(self, namespace: Namespace) -> None:
        self._link_store = get_link_store(namespace)

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


SUGGESTER_STORE: dict[Namespace, LinkSuggester] = {}


def get_link_suggester(namespace: Namespace) -> LinkSuggester:
    res = SUGGESTER_STORE.get(namespace)
    if res is None:
        res = create_link_suggester(namespace)
        SUGGESTER_STORE[namespace] = res
    return res


RandomSuggestModule = TypedDict('RandomSuggestModule', {
    "name": Literal["random"],
})
SuggestModule = RandomSuggestModule


def create_link_suggester(namespace: Namespace) -> LinkSuggester:
    sobj = namespace.get_suggest_module()
    if sobj["name"] == "random":
        from system.suggest.random import RandomLinkSuggester
        return RandomLinkSuggester(namespace)
    raise ValueError(f"unknown link suggester: {sobj}")
