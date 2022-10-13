import time
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

import pandas as pd

from effects.dedicated import (
    AddOp,
    AndOp,
    Branch,
    EqOp,
    OrOp,
    RedisFn,
    RootSet,
    RootValue,
    Script,
    ToJSON,
)
from effects.effects import EffectDependent
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

        def compute_call(
                obj: EffectDependent[PLink, List[str], RLink],
                parents: Tuple[ValueRootRedisType[RLink, float]],
                _: RLink,
                key: PLink) -> None:
            last, = parents
            obj.set_value(
                key,
                list(last.get_range_keys(key_children("vlast", key))))

        self.r_call: ListDependentRedisType[PLink, RLink] = \
            ListDependentRedisType(
                "link",
                key_parent_constructor("vcall"),
                (self.r_last,),
                compute_call,
                lambda pkey: PLink(
                    vote_type=pkey.vote_type, parent=pkey.parent),
                2.0 * dmul)

        # all parents for a given child

        def compute_pall(
                obj: EffectDependent[CLink, List[str], RLink],
                parents: Tuple[ValueRootRedisType[RLink, float]],
                _: RLink,
                key: CLink) -> None:
            last, = parents
            obj.set_value(
                key,
                list(last.get_range_keys(*key_parents("vlast", key))))

        self.r_pall: ListDependentRedisType[CLink, RLink] = \
            ListDependentRedisType(
                "link",
                key_child_constructor("vpall"),
                (self.r_last,),
                compute_pall,
                lambda pkey: CLink(vote_type=pkey.vote_type, child=pkey.child),
                2.0 * dmul)

        # sorted lists by score

        self.r_call_sorted: Dict[
            ScorerName, ListDependentRedisType[PLink, PLink]] = {}
        self.r_pall_sorted: Dict[
            ScorerName, ListDependentRedisType[CLink, CLink]] = {}
        self.r_user_sorted: Dict[
            ScorerName, ListDependentRedisType[str, str]] = {}

        def add_scorer_dependent_types(scorer: Scorer) -> None:
            sname = scorer.name()

            # all children for a given parent sorted with score

            def compute_call_sorted(
                    obj: EffectDependent[PLink, List[str], PLink],
                    parents: Tuple[ListDependentRedisType[PLink, RLink]],
                    pkey: PLink,
                    key: PLink) -> None:
                call, = parents
                now = now_ts()
                links = sorted(
                    (
                        self.get_link(pkey.parent, MHash.parse(child))
                        for child in call.get_value(pkey, [])
                    ),
                    key=lambda link: scorer.get_score(link, now),
                    reverse=True)
                obj.set_value(
                    key, [link.get_child().to_parseable() for link in links])

            self.r_call_sorted[sname] = ListDependentRedisType(
                "link",
                key_parent_constructor(f"scall:{sname}"),
                (self.r_call,),
                compute_call_sorted,
                lambda pkey: pkey,
                2.0 * dmul)

            # all parents for a given child sorted with score

            def compute_pall_sorted(
                    obj: EffectDependent[CLink, List[str], CLink],
                    parents: Tuple[ListDependentRedisType[CLink, RLink]],
                    pkey: CLink,
                    key: CLink) -> None:
                pall, = parents
                now = now_ts()
                links = sorted(
                    (
                        self.get_link(MHash.parse(parent), pkey.child)
                        for parent in pall.get_value(pkey, [])
                    ),
                    key=lambda link: scorer.get_score(link, now),
                    reverse=True)
                obj.set_value(
                    key, [link.get_parent().to_parseable() for link in links])

            self.r_pall_sorted[sname] = ListDependentRedisType(
                "link",
                key_child_constructor(f"spall:{sname}"),
                (self.r_pall,),
                compute_pall_sorted,
                lambda pkey: pkey,
                2.0 * dmul)

            # all links created by a user sorted with score

            def compute_user_sorted(
                    obj: EffectDependent[str, List[str], str],
                    parents: Tuple[SetRootRedisType[str]],
                    pkey: str,
                    key: str) -> None:
                user_links, = parents
                now = now_ts()

                def to_link(ulink: str) -> Link:
                    rlink = parse_link(VT_UP, ulink)
                    return self.get_link(rlink.parent, rlink.child)

                links = sorted(
                    (
                        to_link(ulink)
                        for ulink in user_links.get_value(pkey, set())
                    ),
                    key=lambda link: scorer.get_score(link, now),
                    reverse=True)
                obj.set_value(
                    key,
                    [
                        parseable_link(link.get_parent(), link.get_child())
                        for link in links
                    ])

            self.r_user_sorted[sname] = ListDependentRedisType(
                "link",
                lambda user: f"suserlinks:{sname}:{user}",
                (self.r_user_links,),
                compute_user_sorted,
                lambda pkey: pkey,
                4.0 * dmul)

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
            conn=self._conn)

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

        main = Branch(EqOp(RedisFn("SISMEMBER", r_voted, user_id), 0))
        script.add_stmt(main)
        mseq = main.get_success()
        mseq.add_stmt(RedisFn("SADD", r_voted, user_id).as_stmt())

        total_sum = AddOp(OrOp(RedisFn("GET", r_total), 0), weighted_value)
        mseq.add_stmt(RedisFn("SET", r_total, total_sum).as_stmt())

        daily_sum = AddOp(OrOp(RedisFn("GET", r_daily), 0), weighted_value)
        mseq.add_stmt(RedisFn("SET", r_daily, daily_sum).as_stmt())

        user_new_value = Branch(EqOp(RedisFn("EXISTS", r_user), 0))
        mseq.add_stmt(user_new_value)
        user_new_value.get_success().add_stmt(
            RedisFn("SET", r_user, ToJSON(user_id)).as_stmt())

        first_new_value = Branch(EqOp(RedisFn("EXISTS", r_first), 0))
        mseq.add_stmt(first_new_value)
        fnv_seq = first_new_value.get_success()
        fnv_seq.add_stmt(RedisFn("SET", r_first, now).as_stmt())
        fnv_seq.add_stmt(is_new.assign(True))

        mseq.add_stmt(RedisFn("SET", r_last, now).as_stmt())

        is_user_link = Branch(AndOp(is_new, EqOp(vote_type, VT_UP)))
        mseq.add_stmt(is_user_link)
        is_user_link.get_success().add_stmt(
            RedisFn("SADD", r_user_links, plink).as_stmt())

        return script

    def settle_all(self) -> float:
        start_time = time.monotonic()
        self.r_call.settle_all()
        self.r_pall.settle_all()
        for scall in self.r_call_sorted.values():
            scall.settle_all()
        for spall in self.r_pall_sorted.values():
            spall.settle_all()
        for suser in self.r_user_sorted.values():
            suser.settle_all()
        return time.monotonic() - start_time

    @staticmethod
    def valid_scorers() -> List[Scorer]:
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
