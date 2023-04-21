from typing import Iterable, Literal, TypedDict

import pandas as pd

from system.links.link import Link, RLink, VoteType
from system.links.scorer import get_scorer, Scorer
from system.msgs.message import MHash
from system.namespace.module import ModuleBase
from system.namespace.namespace import ModuleName, Namespace
from system.users.store import get_user_store, UserStore
from system.users.user import User


SerUser = TypedDict('SerUser', {
    "kind": Literal["user"],
    "link": RLink,
    "user": User,
})
SerUserLinks = TypedDict('SerUserLinks', {
    "kind": Literal["user_links"],
    "user": User,
    "links": list[RLink],
})
SerVoted = TypedDict('SerVoted', {
    "kind": Literal["voted"],
    "link": RLink,
    "users": list[User],
})
SerTotal = TypedDict('SerTotal', {
    "kind": Literal["total"],
    "link": RLink,
    "total": float,
})
SerDaily = TypedDict('SerDaily', {
    "kind": Literal["daily"],
    "link": RLink,
    "daily": float,
})
SerFirst = TypedDict('SerFirst', {
    "kind": Literal["first"],
    "link": RLink,
    "first": float,
})
SerLast = TypedDict('SerLast', {
    "kind": Literal["last"],
    "link": RLink,
    "last": float,
})
LinkSer = (
    SerUser
    | SerUserLinks
    | SerVoted
    | SerTotal
    | SerDaily
    | SerFirst
    | SerLast
)


class LinkStore(ModuleBase):
    @staticmethod
    def module_name() -> ModuleName:
        return "links"

    @staticmethod
    def valid_scorers() -> list[Scorer]:
        return [
            get_scorer("best"),
            get_scorer("top"),
            get_scorer("new"),
        ]

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

    def get_all_children_count(self, parent: MHash, now: pd.Timestamp) -> int:
        return len(list(self.get_all_children(parent, now)))

    def get_all_parents_count(self, child: MHash, now: pd.Timestamp) -> int:
        return len(list(self.get_all_parents(child, now)))

    def get_all_user_count(self, user: User) -> int:
        return len(list(self.get_all_user_links(user)))

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

    def get_all_totals(self) -> Iterable[tuple[float, VoteType, Link]]:
        raise NotImplementedError()

    def enumerate_votes(
            self,
            user_store: UserStore,
            *,
            progress_bar: bool) -> Iterable[LinkSer]:
        raise NotImplementedError()

    def do_parse_vote_fragment(
            self, link_ser: LinkSer, now: pd.Timestamp | None) -> None:
        raise NotImplementedError()

    def from_namespace(
            self,
            own_namespace: Namespace,
            other_namespace: Namespace,
            *,
            progress_bar: bool) -> None:
        other_links = get_link_store(other_namespace)
        other_users = get_user_store(own_namespace)
        now = None
        for link_ser in other_links.enumerate_votes(
                other_users, progress_bar=progress_bar):
            self.do_parse_vote_fragment(link_ser, now=now)


LINK_STORE: dict[Namespace, LinkStore] = {}


def get_link_store(namespace: Namespace) -> LinkStore:
    res = LINK_STORE.get(namespace)
    if res is None:
        res = create_link_store(namespace)
        LINK_STORE[namespace] = res
    return res


RedisLinkModule = TypedDict('RedisLinkModule', {
    "name": Literal["redis"],
    "conn": str,
})
ColdLinkModule = TypedDict('ColdLinkModule', {
    "name": Literal["cold"],
    "keep_alive": float,
})
LinkModule = RedisLinkModule | ColdLinkModule


def create_link_store(namespace: Namespace) -> LinkStore:
    lobj = namespace.get_link_module()
    if lobj["name"] == "redis":
        from system.links.redisstore import RedisLinkStore

        return RedisLinkStore(
            namespace.get_redis_key("linkstore", lobj["conn"]))
    if lobj["name"] == "cold":
        from system.links.cold import ColdLinkStore

        return ColdLinkStore(
            namespace.get_module_root("links"), keep_alive=lobj["keep_alive"])
    raise ValueError(f"unknown link store: {lobj}")
