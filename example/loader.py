
import collections
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from example.action import (
    Action,
    is_link_action,
    is_message_action,
    parse_action,
)
from misc.io import open_read
from misc.util import from_timestamp, read_jsonl
from system.links.link import parse_vote_type
from system.links.store import LinkStore
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore
from system.users.store import UserStore
from system.users.user import User


def actions_from_file(fname: str) -> Iterable[Action]:
    with open_read(fname, text=True) as fin:
        for obj in read_jsonl(fin):
            yield parse_action(obj)


TYPE_CONVERTER = {
    "up": "up",
    "down": "down",
}


def interpret_action(
        action: Action,
        *,
        message_store: MessageStore,
        link_store: LinkStore,
        user_store: UserStore,
        now: pd.Timestamp,
        hash_lookup: Dict[str, MHash],
        lookup_buffer: DefaultDict[str, List[Action]],
        ) -> Optional[Tuple[str, bool]]:
    ref_id = action["ref_id"]
    if is_link_action(action):
        assert action["link"] is not None
        link = action["link"]
        own_hash = hash_lookup.get(ref_id)
        if own_hash is None:
            lookup_buffer[ref_id].append(action)
            return None
        parent_ref = link["parent_ref"]
        parent_hash = hash_lookup.get(parent_ref)
        if parent_hash is None:
            lookup_buffer[parent_ref].append(action)
            return None
        cur_link = link_store.get_link(parent_hash, own_hash)
        user_name = link.get("user_name", "__no_user__")
        assert user_name is not None  # NOTE: mypy bug?
        user_id = user_store.get_id_from_name(user_name)
        try:
            user = user_store.get_user_by_id(user_id)
        except KeyError:
            user = User(user_name, {
                "can_create_topic": False,
            })
            user_store.store_user(user)
            print(f"create user {user.get_name()}")
        created_ts = from_timestamp(link["created_utc"])
        for vname, vcount in link["votes"].items():
            vtype = parse_vote_type(TYPE_CONVERTER.get(vname, "honor"))
            prev_votes = cur_link.get_votes(vtype)
            total_votes = int(prev_votes.get_total_votes())
            casts = vcount - total_votes
            if casts <= 0:
                continue
            for _ in range(casts):
                cur_link.add_vote(
                    user_store,
                    vtype,
                    user,
                    now if total_votes > 0 else created_ts)
        return None
    if is_message_action(action):
        assert action["message"] is not None
        message = action["message"]
        text = message["text"]
        is_topic = False
        if text.startswith("r/"):
            text = f"t/{text[2:]}"
            is_topic = True
        msg = Message(msg=text)
        if is_topic:
            mhash = message_store.add_topic(msg)
        else:
            mhash = message_store.write_message(msg)
        hash_lookup[ref_id] = mhash
        return ref_id, is_topic
    raise ValueError(f"unknown action: {action['kind']}")


def process_actions(
        actions: Iterable[Action],
        *,
        message_store: MessageStore,
        link_store: LinkStore,
        user_store: UserStore,
        now: pd.Timestamp,
        hash_lookup: Dict[str, MHash],
        lookup_buffer: DefaultDict[str, List[Action]],
        topic_counts: DefaultDict[str, int]) -> pd.Timestamp:
    for action in actions:
        ref = interpret_action(
            action,
            message_store=message_store,
            link_store=link_store,
            user_store=user_store,
            now=now,
            hash_lookup=hash_lookup,
            lookup_buffer=lookup_buffer)
        if ref is not None:
            ref_id, is_topic = ref
            if is_topic:
                prev_counts = max(topic_counts.values())
                topic_counts[ref_id] += 1
                if topic_counts[ref_id] > prev_counts:
                    now += pd.Timedelta("1d")
                    print(f"advance date to {now}")
            lb_actions = lookup_buffer.pop(ref_id)
            if lb_actions:
                now = process_actions(
                    lb_actions,
                    message_store=message_store,
                    link_store=link_store,
                    user_store=user_store,
                    now=now,
                    hash_lookup=hash_lookup,
                    lookup_buffer=lookup_buffer,
                    topic_counts=topic_counts)
    return now


def process_action_file(
        fname: str,
        *,
        message_store: MessageStore,
        link_store: LinkStore,
        user_store: UserStore,
        now: pd.Timestamp) -> None:
    hash_lookup: Dict[str, MHash] = {}
    lookup_buffer: DefaultDict[str, List[Action]] = \
        collections.defaultdict(list)
    topic_counts: DefaultDict[str, int] = \
        collections.defaultdict(lambda: 0)
    process_actions(
        actions_from_file(fname),
        message_store=message_store,
        link_store=link_store,
        user_store=user_store,
        now=now,
        hash_lookup=hash_lookup,
        lookup_buffer=lookup_buffer,
        topic_counts=topic_counts)
