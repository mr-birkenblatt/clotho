from typing import Iterable, Optional, Tuple

import pandas as pd

from misc.redis import ObjectRedis
from misc.util import from_timestamp, to_timestamp
from system.links.link import Link, parse_vote_type, Votes, VoteType, VT_UP
from system.links.store import LinkStore
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


class RedisLink(Link):
    def __init__(
            self, redis: ObjectRedis, parent: MHash, child: MHash) -> None:
        self._r = redis
        self._parent = parent
        self._child = child
        self._user: Optional[User] = None

    def get_parent(self) -> MHash:
        return self._parent

    def get_child(self) -> MHash:
        return self._child

    def _construct_key(self, vote_type: VoteType) -> str:
        return (
            f"{vote_type}:{self._parent.to_parseable()}:"
            f"{self._child.to_parseable()}")

    @staticmethod
    def parse_key(key: str) -> Tuple[VoteType, MHash, MHash]:
        vtype, parent, child = key.split(":", 2)
        return (
            parse_vote_type(vtype),
            MHash.parse(parent),
            MHash.parse(child),
        )

    def get_user(self, user_store: UserStore) -> Optional[User]:
        if self._user is not None:
            return self._user
        user_id = self._r.obj_get("user", self._construct_key(VT_UP))
        if user_id is None:
            return None
        res = user_store.get_user_by_id(user_id)
        self._user = res
        return res

    def get_votes(self, vote_type: VoteType) -> Votes:
        key = self._construct_key(vote_type)
        vtotal = float(self._r.obj_get("vtotal", key, default=0.0))
        vdaily = float(self._r.obj_get("vdaily", key, default=0.0))
        vfirst = self._r.obj_get("vfirst", key)
        vlast = self._r.obj_get("vlast", key)
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
        # FIXME: remove lock / make atomic
        with self._r.get_lock("votes"):
            key = self._construct_key(vote_type)
            votes = self.get_votes(vote_type)
            weighted_value = who.get_weighted_vote(self.get_user(user_store))
            self._r.obj_put(
                "vtotal", key, votes.get_total_votes() + weighted_value)
            self._r.obj_put(
                "vdaily",
                key,
                votes.get_adjusted_daily_votes(now) + weighted_value)
            self._r.obj_put_nx("user", key, who.get_id())
            self._r.obj_put_nx("vfirst", key, to_timestamp(now))
            self._r.obj_put("vlast", key, to_timestamp(now))


class RedisLinkStore(LinkStore):
    def __init__(self) -> None:
        self._r = ObjectRedis("link")

    def get_link(self, parent: MHash, child: MHash) -> Link:
        return RedisLink(self._r, parent, child)

    def get_all_children(self, parent: MHash) -> Iterable[Link]:
        for child in self._r.obj_partial_keys(
                f"vfirst:{VT_UP}:{parent.to_parseable()}:"):
            yield self.get_link(parent, MHash.parse(child))

    def get_all_parents(self, child: MHash) -> Iterable[Link]:
        for key in self._r.obj_partial_keys(f"vfirst:{VT_UP}:"):
            parent, cur_child = key.split(":", 1)
            if MHash.parse(cur_child) != child:
                continue
            yield self.get_link(MHash.parse(parent), child)

    def get_all_user_links(self, user: User) -> Iterable[Link]:
        user_id = user.get_id()
        for key, value in self._r.obj_dict("user").items():
            vtype, parent, child = RedisLink.parse_key(key)
            if vtype != VT_UP:
                continue
            if value != user_id:
                continue
            yield self.get_link(parent, child)
