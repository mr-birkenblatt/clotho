import time
from typing import Callable, Iterable, NamedTuple

import pandas as pd

from effects.dedicated import RedisFn, RootSet, RootValue, Script
from effects.redis import (
    ListDependentRedisType,
    SetRootRedisType,
    ValueRootRedisType,
)
from misc.redis import RedisConnection
from misc.util import from_timestamp, now_ts, to_timestamp
from system.links.link import Link, Votes, VoteType, VT_UP
from system.links.scorer import get_scorer, Scorer, ScorerName
from system.links.store import LinkStore
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


DELAY_MULTIPLIER = 1.0


def set_delay_multiplier(mul: float) -> None:
    global DELAY_MULTIPLIER

    DELAY_MULTIPLIER = mul


def get_delay_multiplier() -> float:
    return DELAY_MULTIPLIER


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


def parseable_link(parent: MHash, child: MHash) -> str:
    return f"{parent.to_parseable()}:{child.to_parseable()}"


def parse_link(vote_type: VoteType, link: str) -> RLink:
    parent, child = link.split(":", 1)
    return RLink(
        vote_type=vote_type,
        parent=MHash.parse(parent),
        child=MHash.parse(child))


def key_children(prefix: str, link: PLink) -> str:
    return f"{prefix}:{link.vote_type}:{link.parent.to_parseable()}:"


def key_parents(prefix: str, link: CLink) -> tuple[str, str]:
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
        self._user: User | None = None

    def get_parent(self) -> MHash:
        return self._parent

    def get_child(self) -> MHash:
        return self._child

    def get_user(self, user_store: UserStore) -> User | None:
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

        def get_voters(user_store: UserStore) -> Iterable[User]:
            return (
                user_store.get_user_by_id(user_id)
                for user_id in store.r_voted.get_value(key, set())
            )

        return Votes(vote_type, vdaily, vtotal, first, last, get_voters)

    def add_vote(
            self,
            user_store: UserStore,
            vote_type: VoteType,
            who: User,
            now: pd.Timestamp) -> None:
        store = self._s
        key = RLink(
            vote_type=vote_type, parent=self._parent, child=self._child)
        weighted_value = who.get_weighted_vote(self.get_user(user_store))
        nows = to_timestamp(now)
        store.add_vote(key, who, vote_type, weighted_value, nows)


class RedisLinkStore(LinkStore):
    def __init__(self) -> None:
        dmul = get_delay_multiplier()
        self.r_user: ValueRootRedisType[RLink, str] = ValueRootRedisType(
            "link", key_constructor("user"))
        self.r_user_links = SetRootRedisType[str](
            "link", lambda user: f"vuserlinks:{user}")
        self.r_voted: SetRootRedisType[RLink] = SetRootRedisType(
            "link", key_constructor("voted"))
        self.r_total: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vtotal"))
        self.r_daily: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vdaily"))
        self.r_first: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vfirst"))
        self.r_last: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            "link", key_constructor("vlast"))

        # all children for a given parent

        def compute_call(key: PLink) -> None:
            self.r_call.set_value(
                key,
                self.r_last.get_range_keys(key_children("vlast", key)))

        self.r_call: ListDependentRedisType[PLink] = \
            ListDependentRedisType(
                "link",
                key_parent_constructor("vcall"),
                parents=(self.r_last,),
                convert=lambda pkey: PLink(
                    vote_type=pkey.vote_type, parent=pkey.parent),
                effect=compute_call,
                delay=2.0 * dmul)

        # all parents for a given child

        def compute_pall(key: CLink) -> None:
            self.r_pall.set_value(
                key,
                self.r_last.get_range_keys(*key_parents("vlast", key)))

        self.r_pall: ListDependentRedisType[CLink] = \
            ListDependentRedisType(
                "link",
                key_child_constructor("vpall"),
                parents=(self.r_last,),
                convert=lambda pkey: CLink(
                    vote_type=pkey.vote_type, child=pkey.child),
                effect=compute_pall,
                delay=2.0 * dmul)

        # sorted lists by score

        self.r_call_sorted: dict[
            ScorerName, ListDependentRedisType[PLink]] = {}
        self.r_pall_sorted: dict[
            ScorerName, ListDependentRedisType[CLink]] = {}
        self.r_user_sorted: dict[
            ScorerName, ListDependentRedisType[str]] = {}

        def add_scorer_dependent_types(scorer: Scorer) -> None:
            sname = scorer.name()

            # all children for a given parent sorted with score

            def compute_call_sorted(key: PLink) -> None:
                now = now_ts()
                links = sorted(
                    (
                        self.get_link(key.parent, MHash.parse(child))
                        for child in self.r_call.get_value(key, [])
                    ),
                    key=lambda link: scorer.get_score(link, now),
                    reverse=True)
                cur_r_call_sorted.set_value(
                    key, [link.get_child().to_parseable() for link in links])

            cur_r_call_sorted = ListDependentRedisType(
                "link",
                key_parent_constructor(f"scall:{sname}"),
                parents=(self.r_call,),
                convert=lambda pkey: pkey,
                effect=compute_call_sorted,
                delay=2.0 * dmul)
            self.r_call_sorted[sname] = cur_r_call_sorted

            # all parents for a given child sorted with score

            def compute_pall_sorted(key: CLink) -> None:
                now = now_ts()
                links = sorted(
                    (
                        self.get_link(MHash.parse(parent), key.child)
                        for parent in self.r_pall.get_value(key, [])
                    ),
                    key=lambda link: scorer.get_score(link, now),
                    reverse=True)
                cur_r_pall_sorted.set_value(
                    key, [link.get_parent().to_parseable() for link in links])

            cur_r_pall_sorted = ListDependentRedisType(
                "link",
                key_child_constructor(f"spall:{sname}"),
                parents=(self.r_pall,),
                convert=lambda pkey: pkey,
                effect=compute_pall_sorted,
                delay=2.0 * dmul)
            self.r_pall_sorted[sname] = cur_r_pall_sorted

            # all links created by a user sorted with score

            def to_link(ulink: str) -> Link:
                rlink = parse_link(VT_UP, ulink)
                return self.get_link(rlink.parent, rlink.child)

            def compute_user_sorted(key: str) -> None:
                now = now_ts()
                links = sorted(
                    (
                        to_link(ulink)
                        for ulink in self.r_user_links.get_value(key, set())
                    ),
                    key=lambda link: scorer.get_score(link, now),
                    reverse=True)
                cur_r_user_sorted.set_value(
                    key,
                    [
                        parseable_link(link.get_parent(), link.get_child())
                        for link in links
                    ])

            cur_r_user_sorted = ListDependentRedisType(
                "link",
                lambda user: f"suserlinks:{sname}:{user}",
                parents=(self.r_user_links,),
                convert=lambda pkey: pkey,
                effect=compute_user_sorted,
                delay=4.0 * dmul)
            self.r_user_sorted[sname] = cur_r_user_sorted

        for scorer in self.valid_scorers():
            add_scorer_dependent_types(scorer)

        # add_vote lua script
        self._conn = RedisConnection("link")
        self._add_vote = self.create_add_vote_script()

    def add_vote(
            self,
            link: RLink,
            user: User,
            vote_type: VoteType,
            weighted_value: float,
            now: float) -> None:
        user_id = user.get_id()
        self._add_vote.execute(
            args={
                "user_id": user_id,
                "weighted_value": weighted_value,
                "vote_type": vote_type,
                "now": now,
                "plink": parseable_link(link.parent, link.child),
            },
            keys={
                "r_voted": link,
                "r_total": link,
                "r_daily": link,
                "r_user": link,
                "r_first": link,
                "r_last": link,
                "r_user_links": user_id,
            },
            conn=self._conn,
            depth=1)

    def create_add_vote_script(self) -> Script:
        script = Script()
        user_id = script.add_arg("user_id")
        weighted_value = script.add_arg("weighted_value")
        vote_type = script.add_arg("vote_type")
        now = script.add_arg("now")
        plink = script.add_arg("plink")
        r_voted: RootSet[RLink] = script.add_key(
            "r_voted", RootSet(self.r_voted))
        r_total: RootValue[RLink, float] = script.add_key(
            "r_total", RootValue(self.r_total))
        r_daily: RootValue[RLink, float] = script.add_key(
            "r_daily", RootValue(self.r_daily))
        r_user: RootValue[RLink, str] = script.add_key(
            "r_user", RootValue(self.r_user))
        r_first: RootValue[RLink, float] = script.add_key(
            "r_first", RootValue(self.r_first))
        r_last: RootValue[RLink, float] = script.add_key(
            "r_last", RootValue(self.r_last))
        r_user_links: RootSet[str] = script.add_key(
            "r_user_links", RootSet(self.r_user_links))
        is_new = script.add_local(False)

        mseq, _ = script.if_(RedisFn("SISMEMBER", r_voted, user_id).eq(0))
        mseq.add(RedisFn("SADD", r_voted, user_id))

        total_sum = RedisFn("GET", r_total).or_(0) + weighted_value
        mseq.add(RedisFn("SET", r_total, total_sum))

        daily_sum = RedisFn("GET", r_daily).or_(0) + weighted_value
        mseq.add(RedisFn("SET", r_daily, daily_sum))

        user_exists, _ = mseq.if_(RedisFn("EXISTS", r_user).eq(0))
        user_exists.add(RedisFn("SET", r_user, user_id.json()))

        fnv_seq, _ = mseq.if_(RedisFn("EXISTS", r_first).eq(0))
        fnv_seq.add((
            RedisFn("SET", r_first, now),
            is_new.assign(True),
        ))

        mseq.add(RedisFn("SET", r_last, now))

        is_user_link, _ = mseq.if_(is_new.and_(vote_type.eq(VT_UP)))
        is_user_link.add(RedisFn("SADD", r_user_links, plink))

        return script

    def settle_all(self) -> tuple[int, float]:
        start_time = time.monotonic()
        count = 0
        count += self.r_call.settle_all()
        count += self.r_pall.settle_all()
        for scall in self.r_call_sorted.values():
            count += scall.settle_all()
        for spall in self.r_pall_sorted.values():
            count += spall.settle_all()
        for suser in self.r_user_sorted.values():
            count += suser.settle_all()
        return count, time.monotonic() - start_time

    @staticmethod
    def valid_scorers() -> list[Scorer]:
        return [
            get_scorer("best"),
            get_scorer("top"),
            get_scorer("new"),
        ]

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
            limit: int) -> Iterable[Link]:
        for child in self.r_call_sorted[scorer.name()].get_value_range(
                PLink(vote_type=VT_UP, parent=parent),
                offset,
                offset + limit):
            yield self.get_link(parent, MHash.parse(child))

    def get_parents(
            self,
            child: MHash,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> Iterable[Link]:
        for parent in self.r_pall_sorted[scorer.name()].get_value_range(
                CLink(vote_type=VT_UP, child=child),
                offset,
                offset + limit):
            yield self.get_link(MHash.parse(parent), child)

    def get_user_links(
            self,
            user: User,
            *,
            scorer: Scorer,
            now: pd.Timestamp,
            offset: int,
            limit: int) -> Iterable[Link]:
        user_id = user.get_id()
        for link in self.r_user_sorted[scorer.name()].get_value_range(
                user_id, offset, offset + limit):
            rlink = parse_link(VT_UP, link)
            yield self.get_link(rlink.parent, rlink.child)
