import os
from typing import Any, Iterable, NoReturn

import pandas as pd

from misc.cold_writer import ColdAccess
from misc.util import json_compact, json_maybe_read
from system.links.link import Link, parse_vote_type, RLink, VoteType
from system.links.store import LinkSer, LinkStore, SerTotal
from system.msgs.message import MHash
from system.users.store import UserStore
from system.users.user import User


def not_supported() -> NoReturn:
    raise RuntimeError(
        "method is not supported in cold storage "
        "because it would require a full scan")


def parse_link(value: Any) -> RLink:
    return RLink(
        vote_type=parse_vote_type(value["vote_type"]),
        parent=MHash.parse(value["parent"]),
        child=MHash.parse(value["child"]))


def total_from_json(line: str) -> SerTotal | None:
    obj = json_maybe_read(line)
    assert obj is not None
    kind = obj["kind"]
    if kind == "total":
        return {
            "kind": "total",
            "link": parse_link(obj["link"]),
            "total": float(obj["total"]),
        }
    return None


def from_json(line: str, user_store: UserStore) -> LinkSer:
    obj = json_maybe_read(line)
    assert obj is not None
    kind = obj["kind"]
    if kind == "user":
        return {
            "kind": "user",
            "link": parse_link(obj["link"]),
            "user": user_store.get_user_by_id(obj["user"]),
        }
    if kind == "user_links":
        return {
            "kind": "user_links",
            "user": user_store.get_user_by_id(obj["user"]),
            "links": [
                parse_link(link)
                for link in obj["links"]
            ],
        }
    if kind == "voted":
        return {
            "kind": "voted",
            "link": parse_link(obj["link"]),
            "users": [
                user_store.get_user_by_id(user_id)
                for user_id in obj["users"]
            ],
        }
    if kind == "total":
        return {
            "kind": "total",
            "link": parse_link(obj["link"]),
            "total": float(obj["total"]),
        }
    if kind == "daily":
        return {
            "kind": "daily",
            "link": parse_link(obj["link"]),
            "daily": float(obj["daily"]),
        }
    if kind == "first":
        return {
            "kind": "first",
            "link": parse_link(obj["link"]),
            "first": float(obj["first"]),
        }
    if kind == "last":
        return {
            "kind": "last",
            "link": parse_link(obj["link"]),
            "last": float(obj["last"]),
        }
    raise ValueError(f"could not parse: '{line}'")


def to_json(link_ser: LinkSer) -> bytes:

    def prepare(key: str, value: Any) -> Any:
        if key == "kind":
            return value
        if key in ("total", "daily", "first", "last"):
            return value
        if key == "link":
            link: RLink = value
            return {
                "vote_type": link.vote_type,
                "parent": link.parent.to_parseable(),
                "child": link.child.to_parseable(),
            }
        if key == "user":
            user: User = value
            return user.get_id()
        if key == "links":
            links: list[RLink] = value
            return [
                {
                    "vote_type": link.vote_type,
                    "parent": link.parent.to_parseable(),
                    "child": link.child.to_parseable(),
                }
                for link in links
            ]
        if key == "users":
            users: list[User] = value
            return [
                user.get_id()
                for user in users
            ]
        raise ValueError(f"unknown key {key} value {value}")

    return json_compact({
        key: prepare(key, value)
        for key, value in link_ser.items()
    })


class ColdLinkStore(LinkStore):
    def __init__(self, root: str, *, keep_alive: float) -> None:
        super().__init__()
        self._links = ColdAccess(
            os.path.join(root, "links.zip"), keep_alive=keep_alive)

    def enumerate_votes(
            self,
            user_store: UserStore,
            *,
            progress_bar: bool) -> Iterable[LinkSer]:
        for line in self._links.enumerate_lines():
            if not line:
                continue
            yield from_json(line, user_store)

    def get_all_totals(self) -> Iterable[tuple[float, VoteType, Link]]:
        for line in self._links.enumerate_lines():
            if not line:
                continue
            mser = total_from_json(line)
            if mser is None:
                continue
            rlink = mser["link"]
            yield (
                mser["total"],
                rlink.vote_type,
                Link(self, rlink.parent, rlink.child),
            )

    def do_parse_vote_fragment(
            self, link_ser: LinkSer, now: pd.Timestamp | None) -> None:
        self._links.write_line(to_json(link_ser).decode("utf-8"))

    def get_user_id(self, link: RLink) -> str | None:
        not_supported()

    def get_vote_total(self, link: RLink) -> float:
        not_supported()

    def get_vote_daily(self, link: RLink) -> float:
        not_supported()

    def get_vote_first(self, link: RLink) -> float | None:
        not_supported()

    def get_vote_last(self, link: RLink) -> float | None:
        not_supported()

    def has_voted(self, link: RLink, user: User) -> bool:
        not_supported()

    def get_voters(self, link: RLink, user_store: UserStore) -> Iterable[User]:
        not_supported()

    def add_vote(
            self,
            link: RLink,
            user: User,
            vote_type: VoteType,
            weighted_value: float,
            now: pd.Timestamp) -> None:
        not_supported()

    def remove_vote(
            self,
            link: RLink,
            user: User,
            weighted_value: float,
            now: pd.Timestamp) -> None:
        not_supported()

    def get_all_children(
            self, parent: MHash, now: pd.Timestamp) -> Iterable[Link]:
        not_supported()

    def get_all_parents(
            self, child: MHash, now: pd.Timestamp) -> Iterable[Link]:
        not_supported()

    def get_all_user_links(self, user: User) -> Iterable[Link]:
        not_supported()
