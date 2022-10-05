from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import pandas as pd

from effects.effects import EffectDependent
from effects.redis import (
    ListDependentRedisType,
    SetRootRedisType,
    ValueDependentRedisType,
    ValueRootRedisType,
)
from misc.util import from_timestamp, to_timestamp
from system.links.link import Link, Votes, VoteType, VT_UP
from system.links.scorer import Scorer
from system.links.store import LinkStore
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


RLink = NamedTuple('RLink', [
    ("vote_type", VoteType),
    ("parent", MHash),
    ("child", MHash),
])
PLink = NamedTuple('PLink', [
    ("vote_type", VoteType),
    ("parent", MHash),
])
CLink = NamedTuple('CLink', [
    ("vote_type", VoteType),
    ("child", MHash),
])


def parseable_link(link: RLink) -> str:
    return f"{link.parent.to_parseable()}:{link.child.to_parseable()}"


def parse_link(vote_type: VoteType, link: str) -> RLink:
    parent, child = link.split(":", 1)
    return RLink(
        vote_type=vote_type,
        parent=MHash.parse(parent),
        child=MHash.parse(child))


def key_children(prefix: str, link: PLink) -> str:
    return f"{prefix}:{link.vote_type}:{link.parent.to_parseable()}:"


def key_parents(prefix: str, link: CLink) -> Tuple[str, str]:
    return (
        f"{prefix}:{link.vote_type}:",
        f":{link.child.to_parseable()}",
    )


def key_parent_constructor(prefix: str) -> Callable[[PLink], str]:

    def construct_key(link: PLink) -> str:
        return (
            f"{prefix}:{link.vote_type}:"
            f"{link.parent.to_parseable()}")

    return construct_key


def key_child_constructor(prefix: str) -> Callable[[CLink], str]:

    def construct_key(link: CLink) -> str:
        return (
            f"{prefix}:{link.vote_type}:"
            f"{link.child.to_parseable()}")

    return construct_key


def key_constructor(prefix: str) -> Callable[[RLink], str]:

    def construct_key(link: RLink) -> str:
        return (
            f"{prefix}:{link.vote_type}:"
            f"{link.parent.to_parseable()}:"
            f"{link.child.to_parseable()}")

    return construct_key


class RedisLink(Link):
    def __init__(
            self,
            store: 'RedisLinkStore',
            parent: MHash,
            child: MHash) -> None:
        self._s = store
        self._parent = parent
        self._child = child
        self._user: Optional[User] = None

    def get_parent(self) -> MHash:
        return self._parent

    def get_child(self) -> MHash:
        return self._child

    def get_user(self, user_store: UserStore) -> Optional[User]:
        if self._user is not None:
            return self._user
        store = self._s
        key = RLink(
            vote_type=VT_UP, parent=self._parent, child=self._child)
        user_id = store.r_user.maybe_get_value(key)
        if user_id is None:
            return None
        res = user_store.get_user_by_id(user_id)
        self._user = res
        return res

    def get_votes(self, vote_type: VoteType) -> Votes:
        store = self._s
        key = RLink(
            vote_type=vote_type, parent=self._parent, child=self._child)
        vtotal = store.r_total.get_value(key, default=0.0)
        vdaily = store.r_daily.get_value(key, default=0.0)
        vfirst = store.r_first.maybe_get_value(key)
        vlast = store.r_last.maybe_get_value(key)
        if vfirst is None:
            first = None
        else:
            first = from_timestamp(float(vfirst))
        if vlast is None:
            last = None
        else:
            last = from_timestamp(float(vlast))
        return Votes(vote_type, vdaily, vtotal, first, last)

    def add_vote(
            self,
            user_store: UserStore,
            vote_type: VoteType,
            who: User,
            now: pd.Timestamp) -> None:
        store = self._s
        key = RLink(
            vote_type=vote_type, parent=self._parent, child=self._child)
        # FIXME: make atomic
        user_id = who.get_id()
        if store.r_voted.add_value(key, user_id):
            return
        votes = self.get_votes(vote_type)
        weighted_value = who.get_weighted_vote(self.get_user(user_store))
        store.r_total.set_value(key, votes.get_total_votes() + weighted_value)
        store.r_daily.set_value(
            key, votes.get_adjusted_daily_votes(now) + weighted_value)
        store.r_user.do_set_new_value(key, user_id)
        store.r_last.set_value(key, to_timestamp(now))


class RedisLinkStore(LinkStore):
    def __init__(self) -> None:
        self.r_user: ValueRootRedisType[RLink, str] = ValueRootRedisType(
            "link", key_constructor("user"))
        self.r_user_links = SetRootRedisType[str](
            "link", lambda user: f"userlinks:{user}")
        self.r_voted: SetRootRedisType[RLink] = SetRootRedisType(
            "link", key_constructor("voted"))
        self.r_total: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vtotal"))
        self.r_daily: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vdaily"))
        self.r_last: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vlast"))

        def compute_first(
                obj: EffectDependent[RLink, float, RLink],
                parents: Tuple[
                    ValueRootRedisType[RLink, float],
                    ValueRootRedisType[RLink, str]],
                key: RLink) -> None:
            last, user = parents
            val = last.maybe_get_value(key)
            if val is None:
                return
            is_new = obj.set_new_value(key, val)
            if is_new and key.vote_type == VT_UP:
                user_id = user.maybe_get_value(key)
                if user_id is not None:
                    self.r_user_links.add_value(user_id, parseable_link(key))

        self.r_first: ValueDependentRedisType[RLink, float, RLink] = \
            ValueDependentRedisType(
                "link",
                key_constructor("vfirst"),
                (self.r_last, self.r_user),
                compute_first,
                5.0)

        def compute_call(
                obj: EffectDependent[PLink, List[str], RLink],
                parents: Tuple[ValueRootRedisType[RLink, float]],
                key: RLink) -> None:
            last, = parents
            pkey = PLink(vote_type=key.vote_type, parent=key.parent)
            obj.set_value(
                pkey,
                list(last.get_range_keys(key_children("vlast", pkey))))

        self.r_call: ListDependentRedisType[PLink, RLink] = \
            ListDependentRedisType(
                "link",
                key_parent_constructor("vcall"),
                (self.r_last,),
                compute_call,
                2.0)

        def compute_pall(
                obj: EffectDependent[CLink, List[str], RLink],
                parents: Tuple[ValueRootRedisType[RLink, float]],
                key: RLink) -> None:
            last, = parents
            ckey = CLink(vote_type=key.vote_type, child=key.child)
            obj.set_value(
                ckey,
                list(last.get_range_keys(*key_parents("vlast", ckey))))

        self.r_pall: ListDependentRedisType[CLink, RLink] = \
            ListDependentRedisType(
                "link",
                key_child_constructor("vpall"),
                (self.r_last,),
                compute_pall,
                2.0)

    def get_link(self, parent: MHash, child: MHash) -> Link:
        return RedisLink(self, parent, child)

    def get_all_children(self, parent: MHash) -> Iterable[Link]:
        for child in self.r_call.get_value(
                PLink(vote_type=VT_UP, parent=parent), default=[]):
            yield self.get_link(parent, MHash.parse(child))

    def get_all_parents(self, child: MHash) -> Iterable[Link]:
        for parent in self.r_pall.get_value(
                CLink(vote_type=VT_UP, child=child), default=[]):
            yield self.get_link(MHash.parse(parent), child)

    def get_all_user_links(self, user: User) -> Iterable[Link]:
        user_id = user.get_id()
        for link in self.r_user_links.get_value(user_id, set()):
            rlink = parse_link(VT_UP, link)
            yield self.get_link(rlink.parent, rlink.child)

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
