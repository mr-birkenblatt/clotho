from typing import Iterable, Literal, TypedDict

import pandas as pd

from system.links.link import Link, RLink, VoteType
from system.links.scorer import Scorer
from system.msgs.message import MHash
from system.namespace.namespace import Namespace
from system.users.store import UserStore
from system.users.user import User


class LinkStore:
    @staticmethod
    def valid_scorers() -> list[Scorer]:
        raise NotImplementedError()

    def get_user_id(self, link: RLink) -> str | None:
        raise NotImplementedError()

    def get_vote_total(self, link: RLink) -> float:
        raise NotImplementedError()

    def get_vote_daily(self, link: RLink) -> float:
        raise NotImplementedError()

    def get_vote_first(self, link: RLink) -> float | None:
        raise NotImplementedError()

    def get_vote_last(self, link: RLink) -> float | None:
        raise NotImplementedError()

    def has_voted(self, link: RLink, user: User) -> bool:
        raise NotImplementedError()

    def get_voters(self, link: RLink, user_store: UserStore) -> Iterable[User]:
        raise NotImplementedError()

    def add_vote(
            self,
            link: RLink,
            user: User,
            vote_type: VoteType,
            weighted_value: float,
            now: pd.Timestamp) -> None:
        raise NotImplementedError()

    def remove_vote(
            self,
            link: RLink,
            user: User,
            weighted_value: float,
            now: pd.Timestamp) -> None:
        raise NotImplementedError()

    def get_all_children(
            self, parent: MHash, now: pd.Timestamp) -> Iterable[Link]:
        raise NotImplementedError()

    def get_all_parents(
            self, child: MHash, now: pd.Timestamp) -> Iterable[Link]:
        raise NotImplementedError()

    def get_all_user_links(self, user: User) -> Iterable[Link]:
        raise NotImplementedError()

    def limit_results(
            self,
            links: Iterable[Link],
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> list[Link]:
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
            limit: int) -> Iterable[Link]:
        return self.limit_results(
            self.get_all_children(parent, now), scorer, now, offset, limit)

    def get_parents(
            self,
            child: MHash,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> Iterable[Link]:
        return self.limit_results(
            self.get_all_parents(child, now), scorer, now, offset, limit)

    def get_user_links(
            self,
            user: User,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> Iterable[Link]:
        return self.limit_results(
            self.get_all_user_links(user), scorer, now, offset, limit)

    def get_link(self, parent: MHash, child: MHash) -> Link:
        return Link(self, parent, child)


LINK_STORE: dict[Namespace, LinkStore] = {}


def get_link_store(namespace: Namespace) -> LinkStore:
    res = LINK_STORE.get(namespace)
    if res is None:
        res = create_link_store(namespace.get_link_module())
        LINK_STORE[namespace] = res
    return res


RedisLinkModule = TypedDict('RedisLinkModule', {
    "name": Literal["redis"],
    "port": int,
    "host": str,
    "passwd": str,
    "prefix": str,
})
LinkModule = RedisLinkModule


def create_link_store(lobj: LinkModule) -> LinkStore:
    if lobj["name"] == "redis":
        from system.links.redisstore import RedisLinkStore
        return RedisLinkStore(
            lobj["host"], lobj["port"], lobj["passwd"], lobj["prefix"])
    raise ValueError(f"unknown link store: {lobj}")
