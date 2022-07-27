from typing import Iterable, List, Optional
import pandas as pd
from misc.env import envload_str

from system.links.link import Link
from system.links.user import User
from system.links.scorer import Scorer
from system.msgs.message import MHash


class LinkStore:
    def get_link(self, parent: MHash, child: MHash) -> Link:
        raise NotImplementedError()

    def get_all_children(self, parent: MHash) -> Iterable[Link]:
        raise NotImplementedError()

    def get_all_parents(self, child: MHash) -> Iterable[Link]:
        raise NotImplementedError()

    def get_all_user_links(self, user: User) -> Iterable[Link]:
        raise NotImplementedError()

    def limit_results(
            self,
            links: Iterable[Link],
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> List[Link]:
        res = sorted(
            links, key=lambda link: scorer.get_score(link, now), reverse=True)
        return res[offset:offset + limit]

    def get_children(
            self,
            parent: MHash,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> List[Link]:
        return self.limit_results(
            self.get_all_children(parent), scorer, now, offset, limit)

    def get_parents(
            self,
            child: MHash,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> List[Link]:
        return self.limit_results(
            self.get_all_parents(child), scorer, now, offset, limit)

    def get_user_links(
            self,
            user: User,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> List[Link]:
        return self.limit_results(
            self.get_all_user_links(user), scorer, now, offset, limit)


DEFAULT_LINK_STORE: Optional[LinkStore] = None


def get_default_link_store() -> LinkStore:
    global DEFAULT_LINK_STORE

    if DEFAULT_LINK_STORE is None:
        DEFAULT_LINK_STORE = get_link_store(
            envload_str("LINK_STORE", default="redis"))
    return DEFAULT_LINK_STORE


def get_link_store(name: str) -> LinkStore:
    if name == "redis":
        from system.links.redisstore import RedisLinkStore
        return RedisLinkStore()
    raise ValueError(f"unknown link store: {name}")
