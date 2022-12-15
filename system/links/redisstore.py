from typing import Callable, Iterable

import pandas as pd

from effects.dedicated import RedisFn, RootSet, RootValue, Script
from effects.redis import (
    ListDependentRedisType,
    SetRootRedisType,
    ValueRootRedisType,
)
from misc.redis import ConfigKey, get_redis_config, RedisConnection
from misc.util import identity, json_compact, json_read, now_ts, to_timestamp
from system.links.link import (
    CLink,
    deserialize_clink,
    deserialize_plink,
    Link,
    PLink,
    RLink,
    serialize_link,
    to_clink,
    to_plink,
    VoteType,
    VT_UP,
)
from system.links.scorer import get_scorer, Scorer, ScorerName
from system.links.store import LinkStore
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


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


class RedisLinkStore(LinkStore):
    def __init__(self, ns_key: ConfigKey) -> None:
        cfg = get_redis_config(ns_key)
        val = "val" if cfg["prefix"] else ""
        obs = "obs"
        pen = "pen"

        self.r_user: ValueRootRedisType[RLink, str] = ValueRootRedisType(
            ns_key, "link", key_constructor("user"))
        self.r_user_links = SetRootRedisType[str](
            ns_key, "link", lambda user: f"vuserlinks:{user}")
        self.r_voted: SetRootRedisType[RLink] = SetRootRedisType(
            ns_key, "link", key_constructor("voted"))
        self.r_total: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            ns_key, "link", key_constructor("vtotal"))
        self.r_daily: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            ns_key, "link", key_constructor("vdaily"))
        self.r_first: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            ns_key, "link", key_constructor("vfirst"))
        self.r_last: ValueRootRedisType[RLink, float] = ValueRootRedisType(
            ns_key, "link", key_constructor("vlast"))

        # all children for a given parent

        def compute_call(key: PLink, now: pd.Timestamp | None) -> None:
            prefix = key_children("vlast", key)
            gap = MHash.parse_size()
            self.r_call.set_value(
                key,
                list(self.r_last.get_range_gap_keys(prefix, gap)),
                now)

        self.r_call: ListDependentRedisType[PLink] = \
            ListDependentRedisType(
                ns_key,
                "link",
                key_parent_constructor("vcall"),
                serialize_link,
                deserialize_plink,
                val,
                obs,
                pen,
                parents=(self.r_last,),
                convert=to_plink,
                effect=compute_call,
                empty=b"")

        # all parents for a given child

        def compute_pall(key: CLink, now: pd.Timestamp | None) -> None:
            prefix, postfix = key_parents("vlast", key)
            gap = MHash.parse_size()
            self.r_pall.set_value(
                key,
                list(self.r_last.get_range_gap_keys(prefix, gap, postfix)),
                now)

        self.r_pall: ListDependentRedisType[CLink] = \
            ListDependentRedisType(
                ns_key,
                "link",
                key_child_constructor("vpall"),
                serialize_link,
                deserialize_clink,
                val,
                obs,
                pen,
                parents=(self.r_last,),
                convert=to_clink,
                effect=compute_pall,
                empty=b"")

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

            def compute_call_sorted(
                    key: PLink, now: pd.Timestamp | None) -> None:
                links = sorted(
                    (
                        self.get_link(key.parent, MHash.parse(child))
                        for child in self.r_call.get_value(key, [], now)
                    ),
                    key=lambda link: scorer.get_score(
                        link, now_ts() if now is None else now),
                    reverse=True)
                cur_r_call_sorted.set_value(
                    key,
                    [link.get_child().to_parseable() for link in links],
                    now)

            cur_r_call_sorted = ListDependentRedisType(
                ns_key,
                "link",
                key_parent_constructor(f"scall:{sname}"),
                serialize_link,
                deserialize_plink,
                val,
                obs,
                pen,
                parents=(self.r_call,),
                convert=lambda pkey: pkey,  # FIXME: use identity
                effect=compute_call_sorted,
                empty=b"")
            self.r_call_sorted[sname] = cur_r_call_sorted

            # all parents for a given child sorted with score

            def compute_pall_sorted(
                    key: CLink, now: pd.Timestamp | None) -> None:
                links = sorted(
                    (
                        self.get_link(MHash.parse(parent), key.child)
                        for parent in self.r_pall.get_value(key, [], now)
                    ),
                    key=lambda link: scorer.get_score(
                        link, now_ts() if now is None else now),
                    reverse=True)
                cur_r_pall_sorted.set_value(
                    key,
                    [link.get_parent().to_parseable() for link in links],
                    now)

            cur_r_pall_sorted = ListDependentRedisType(
                ns_key,
                "link",
                key_child_constructor(f"spall:{sname}"),
                serialize_link,
                deserialize_clink,
                val,
                obs,
                pen,
                parents=(self.r_pall,),
                convert=lambda pkey: pkey,  # FIXME: use identity
                effect=compute_pall_sorted,
                empty=b"")
            self.r_pall_sorted[sname] = cur_r_pall_sorted

            # all links created by a user sorted with score

            def to_link(ulink: str) -> Link:
                rlink = parse_link(VT_UP, ulink)
                return self.get_link(rlink.parent, rlink.child)

            def compute_user_sorted(
                    key: str, now: pd.Timestamp | None) -> None:
                links = sorted(
                    (
                        to_link(ulink)
                        for ulink in self.r_user_links.get_value(key, set())
                    ),
                    key=lambda link: scorer.get_score(
                        link, now_ts() if now is None else now),
                    reverse=True)
                cur_r_user_sorted.set_value(
                    key,
                    [
                        parseable_link(link.get_parent(), link.get_child())
                        for link in links
                    ],
                    now)

            cur_r_user_sorted = ListDependentRedisType(
                ns_key,
                "link",
                lambda user: f"suserlinks:{sname}:{user}",
                json_compact,
                json_read,
                val,
                obs,
                pen,
                parents=(self.r_user_links,),
                convert=identity,
                effect=compute_user_sorted,
                empty=b"")
            self.r_user_sorted[sname] = cur_r_user_sorted

        for scorer in self.valid_scorers():
            add_scorer_dependent_types(scorer)

        # add_vote lua script
        self._conn = RedisConnection(ns_key, "link")
        self._add_vote = self.create_add_vote_script()
        self._remove_vote = self.create_remove_vote_script()

    def add_vote(
            self,
            link: RLink,
            user: User,
            vote_type: VoteType,
            weighted_value: float,
            now: pd.Timestamp) -> None:
        user_id = user.get_id()
        self._add_vote.execute(
            args={
                "user_id": user_id,
                "weighted_value": weighted_value,
                "vote_type": vote_type,
                "now": to_timestamp(now),
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
            now=now,
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

    def remove_vote(
            self,
            link: RLink,
            user: User,
            weighted_value: float,
            now: pd.Timestamp) -> None:
        user_id = user.get_id()
        self._remove_vote.execute(
            args={
                "user_id": user_id,
                "weighted_value": weighted_value,
            },
            keys={
                "r_voted": link,
                "r_total": link,
                "r_daily": link,
            },
            now=now,
            conn=self._conn,
            depth=1)

    def create_remove_vote_script(self) -> Script:
        script = Script()
        user_id = script.add_arg("user_id")
        weighted_value = script.add_arg("weighted_value")
        r_voted: RootSet[RLink] = script.add_key(
            "r_voted", RootSet(self.r_voted))
        r_total: RootValue[RLink, float] = script.add_key(
            "r_total", RootValue(self.r_total))
        r_daily: RootValue[RLink, float] = script.add_key(
            "r_daily", RootValue(self.r_daily))

        mseq, _ = script.if_(RedisFn("SISMEMBER", r_voted, user_id).neq(0))
        mseq.add(RedisFn("SREM", r_voted, user_id))

        total_sum = RedisFn("GET", r_total).or_(0) - weighted_value
        mseq.add(RedisFn("SET", r_total, total_sum))

        daily_sum = RedisFn("GET", r_daily).or_(0) - weighted_value
        mseq.add(RedisFn("SET", r_daily, daily_sum))

        return script

    @staticmethod
    def valid_scorers() -> list[Scorer]:
        return [
            get_scorer("best"),
            get_scorer("top"),
            get_scorer("new"),
        ]

    def get_user_id(self, link: RLink) -> str | None:
        return self.r_user.maybe_get_value(link)

    def get_vote_total(self, link: RLink) -> float:
        return self.r_total.get_value(link, default=0.0)

    def get_vote_daily(self, link: RLink) -> float:
        return self.r_daily.get_value(link, default=0.0)

    def get_vote_first(self, link: RLink) -> float | None:
        return self.r_first.maybe_get_value(link)

    def get_vote_last(self, link: RLink) -> float | None:
        return self.r_last.maybe_get_value(link)

    def has_voted(self, link: RLink, user: User) -> bool:
        return self.r_voted.has_value(link, user.get_id())

    def get_voters(self, link: RLink, user_store: UserStore) -> Iterable[User]:
        return (
            user_store.get_user_by_id(user_id)
            for user_id in self.r_voted.get_value(link, set())
        )

    def get_all_children(
            self, parent: MHash, now: pd.Timestamp) -> Iterable[Link]:
        for child in self.r_call.get_value(
                PLink(vote_type=VT_UP, parent=parent), default=[], when=now):
            yield self.get_link(parent, MHash.parse(child))

    def get_all_parents(
            self, child: MHash, now: pd.Timestamp) -> Iterable[Link]:
        for parent in self.r_pall.get_value(
                CLink(vote_type=VT_UP, child=child), default=[], when=now):
            yield self.get_link(MHash.parse(parent), child)

    def get_all_user_links(self, user: User) -> Iterable[Link]:
        user_id = user.get_id()
        for link in self.r_user_links.get_value(user_id, set()):
            rlink = parse_link(VT_UP, link)
            yield self.get_link(rlink.parent, rlink.child)

    def get_all_children_count(self, parent: MHash, now: pd.Timestamp) -> int:
        return self.r_call.get_size(
            PLink(vote_type=VT_UP, parent=parent), when=now)

    def get_all_parents_count(self, child: MHash, now: pd.Timestamp) -> int:
        return self.r_pall.get_size(
            CLink(vote_type=VT_UP, child=child), when=now)

    def get_all_user_count(self, user: User) -> int:
        user_id = user.get_id()
        return self.r_user_links.get_size(user_id)

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
                offset + limit,
                now):
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
                offset + limit,
                now):
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
                user_id, offset, offset + limit, now):
            rlink = parse_link(VT_UP, link)
            yield self.get_link(rlink.parent, rlink.child)
