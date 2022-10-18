from typing import Callable, cast, get_args, Iterable, Literal, TypedDict

import numpy as np
import pandas as pd

from misc.util import to_timestamp
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


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

LinkResponse = TypedDict('LinkResponse', {
    "parent": str,
    "child": str,
    "user": str | None,
    "first": float,
    "votes": dict[VoteType, float],
})


def parse_vote_type(text: str) -> VoteType:
    if text not in VOTE_TYPES:
        raise ValueError(f"{text} is not a vote type: {VOTE_TYPES}")
    return cast(VoteType, text)


class Votes:
    def __init__(
            self,
            vote_type: VoteType,
            daily: float,
            total: float,
            first: pd.Timestamp | None,
            last: pd.Timestamp | None,
            voters_fn: Callable[[UserStore], Iterable[User]]) -> None:
        assert daily >= 0
        assert total >= 0
        self._type = vote_type
        self._daily = daily
        self._total = total
        self._first = first
        self._last = last
        self._voters: set[User] | None = None
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

    def get_voters(self, user_store: UserStore) -> set[User]:
        voters = self._voters
        if voters is None:
            voters = set(self._voters_fn(user_store))
            self._voters = voters
        return voters


class Link:
    @staticmethod
    def get_vote_types() -> set[VoteType]:
        return VOTE_TYPES

    def get_parent(self) -> MHash:
        raise NotImplementedError()

    def get_child(self) -> MHash:
        raise NotImplementedError()

    def get_user(self, user_store: UserStore) -> User | None:
        raise NotImplementedError()

    def get_votes(self, vote_type: VoteType) -> Votes:
        raise NotImplementedError()

    def add_vote(
            self,
            user_store: UserStore,
            vote_type: VoteType,
            who: User,
            now: pd.Timestamp) -> None:
        raise NotImplementedError()

    def get_response(
            self, user_store: UserStore, now: pd.Timestamp) -> LinkResponse:
        user = self.get_user(user_store)
        user_str = None if user is None else user.get_id()
        first = now
        votes: dict[VoteType, float] = {}
        for vtype in self.get_vote_types():
            cur_vote = self.get_votes(vtype)
            cur_total = cur_vote.get_total_votes()
            if cur_total > 0.0:
                votes[vtype] = cur_total
            cur_first = cur_vote.get_first_vote_time(now)
            if cur_first < first:
                first = cur_first
        return {
            "parent": self.get_parent().to_parseable(),
            "child": self.get_child().to_parseable(),
            "user": user_str,
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
