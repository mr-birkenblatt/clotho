
from typing import Iterable

from misc.env import envload_str
from system.links.link import Link
from system.links.store import get_default_link_store, LinkStore
from system.msgs.message import MHash


class LinkSuggester:
    def __init__(self) -> None:
        self._link_store = get_default_link_store()

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


DEFAULT_LINK_SUGGESTER: LinkSuggester | None = None


def get_default_link_suggester() -> LinkSuggester:
    global DEFAULT_LINK_SUGGESTER

    if DEFAULT_LINK_SUGGESTER is None:
        DEFAULT_LINK_SUGGESTER = get_link_suggester(
            envload_str("LINK_SUGGESTER", default="random"))
    return DEFAULT_LINK_SUGGESTER


def get_link_suggester(name: str) -> LinkSuggester:
    if name == "random":
        from system.suggest.random import RandomLinkSuggester
        return RandomLinkSuggester()
    raise ValueError(f"unknown link suggester: {name}")
