from typing import Iterable

import pandas as pd

from misc.env import envload_str
from system.links.link import Link, RLink, VoteType
from system.links.scorer import Scorer
from system.msgs.message import MHash
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


DEFAULT_LINK_STORE: LinkStore | None = None


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
