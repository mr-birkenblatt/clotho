from typing import (
    Callable,
    cast,
    get_args,
    Iterable,
    Literal,
    NamedTuple,
    TYPE_CHECKING,
    TypedDict,
)

import numpy as np
import pandas as pd

from misc.util import from_timestamp, json_compact, json_read, to_timestamp
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


if TYPE_CHECKING:
    from system.links.redisstore import LinkStore


# view == it showed up
# up == active click on up
# down == active click on down
# ack == following the chain
# skip == moving to next link on same level
# honor == honors
VoteType = Literal["view", "up", "down", "ack", "skip", "honor"]
VOTE_TYPES: set[VoteType] = set(get_args(VoteType))
VT_VIEW: VoteType = "view"
VT_UP: VoteType = "up"
VT_DOWN: VoteType = "down"
VT_ACK: VoteType = "ack"
VT_SKIP: VoteType = "skip"
VT_HONOR: VoteType = "honor"

VoteInfo = TypedDict('VoteInfo', {
    "count": float,
    "uservoted": bool,
})

LinkResponse = TypedDict('LinkResponse', {
    "parent": str,
    "child": str,
    "user": str | None,
    "userid": str | None,
    "first": float,
    "votes": dict[VoteType, VoteInfo],
})


def parse_vote_type(text: str) -> VoteType:
    if text not in VOTE_TYPES:
        raise ValueError(f"{text} is not a vote type: {VOTE_TYPES}")
    return cast(VoteType, text)


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


def to_plink(link: RLink) -> PLink:
    return PLink(link.vote_type, link.parent)


def to_clink(link: RLink) -> CLink:
    return CLink(link.vote_type, link.child)


def serialize_link(link: RLink | PLink | CLink) -> bytes:
    return json_compact({
        key: value.to_parseable() if isinstance(value, MHash) else value
        for key, value in link._asdict().items()
    })


def deserialize_rlink(obj: bytes) -> RLink:
    link = json_read(obj)
    return RLink(
        parse_vote_type(link["vote_type"]),
        MHash.parse(link["parent"]),
        MHash.parse(link["child"]))


def deserialize_plink(obj: bytes) -> PLink:
    link = json_read(obj)
    return PLink(
        parse_vote_type(link["vote_type"]), MHash.parse(link["parent"]))


def deserialize_clink(obj: bytes) -> CLink:
    link = json_read(obj)
    return CLink(
        parse_vote_type(link["vote_type"]), MHash.parse(link["child"]))


class Votes:
    def __init__(
            self,
            vote_type: VoteType,
            daily: float,
            total: float,
            first: pd.Timestamp | None,
            last: pd.Timestamp | None,
            voter_check_fn: Callable[[User], bool],
            voters_fn: Callable[[UserStore], Iterable[User]]) -> None:
        assert daily >= 0
        assert total >= 0
        self._type = vote_type
        self._daily = daily
        self._total = total
        self._first = first
        self._last = last
        self._voters: set[User] | None = None
        self._voter_check_fn = voter_check_fn
        self._voters_fn = voters_fn

    def get_daily_votes(self) -> float:
        return self._daily

    def get_first_vote_time(self, now: pd.Timestamp) -> pd.Timestamp:
        return now if self._first is None else self._first

    def get_last_vote_time(self) -> pd.Timestamp | None:
        return self._last

    def get_total_votes(self) -> float:
        return self._total

    def get_vote_type(self) -> VoteType:
        return self._type

    def get_adjusted_daily_votes(self, now: pd.Timestamp) -> float:
        if self._last is None:
            return 0.0
        diff = (now - self._last) / pd.Timedelta("1d")
        return self._daily * np.exp(-diff)

    def has_user_voted(self, user: User) -> bool:
        return self._voter_check_fn(user)

    def get_voters(self, user_store: UserStore) -> set[User]:
        voters = self._voters
        if voters is None:
            voters = set(self._voters_fn(user_store))
            self._voters = voters
        return voters


class Link:
    def __init__(
            self,
            store: 'LinkStore',
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
        user_id = store.get_user_id(key)
        if user_id is None:
            return None
        res = user_store.get_user_by_id(user_id)
        self._user = res
        return res

    def get_votes(self, vote_type: VoteType) -> Votes:
        store = self._s
        key = RLink(
            vote_type=vote_type, parent=self._parent, child=self._child)
        vtotal = store.get_vote_total(key)
        vdaily = store.get_vote_daily(key)
        vfirst = store.get_vote_first(key)
        vlast = store.get_vote_last(key)
        if vfirst is None:
            first = None
        else:
            first = from_timestamp(float(vfirst))
        if vlast is None:
            last = None
        else:
            last = from_timestamp(float(vlast))

        def check_voter(user: User) -> bool:
            return store.has_voted(key, user)

        def get_voters(user_store: UserStore) -> Iterable[User]:
            return store.get_voters(key, user_store)

        return Votes(
            vote_type, vdaily, vtotal, first, last, check_voter, get_voters)

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
        store.add_vote(key, who, vote_type, weighted_value, now)

    def remove_vote(
            self,
            user_store: UserStore,
            vote_type: VoteType,
            who: User,
            now: pd.Timestamp) -> None:
        store = self._s
        key = RLink(
            vote_type=vote_type, parent=self._parent, child=self._child)
        weighted_value = who.get_weighted_vote(self.get_user(user_store))
        store.remove_vote(key, who, weighted_value, now)

    @staticmethod
    def get_vote_types() -> set[VoteType]:
        return VOTE_TYPES

    def get_response(
            self,
            user_store: UserStore,
            *,
            who: User | None,
            now: pd.Timestamp) -> LinkResponse:
        user = self.get_user(user_store)
        user_str = None if user is None else user.get_name()
        userid_str = None if user is None else user.get_id()
        first = now
        votes: dict[VoteType, VoteInfo] = {}
        for vtype in self.get_vote_types():
            cur_vote = self.get_votes(vtype)
            cur_total = cur_vote.get_total_votes()
            if cur_total > 0.0:
                uservoted = who is not None and cur_vote.has_user_voted(who)
                votes[vtype] = {
                    "count": cur_total,
                    "uservoted": uservoted,
                }
            cur_first = cur_vote.get_first_vote_time(now)
            if cur_first < first:
                first = cur_first
        return {
            "parent": self.get_parent().to_parseable(),
            "child": self.get_child().to_parseable(),
            "user": user_str,
            "userid": userid_str,
            "first": to_timestamp(first),
            "votes": votes,
        }

    def __hash__(self) -> int:
        return hash(self.get_parent()) + 31 * hash(self.get_child())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return (
            self.get_parent() == other.get_parent()
            and self.get_child() == other.get_child())

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return f"{self.get_parent()} -> {self.get_child()}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.__str__()}]"
