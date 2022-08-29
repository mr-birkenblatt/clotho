
import collections
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

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
        roots: Set[str],
        hash_lookup: Dict[str, MHash],
        lookup_buffer: DefaultDict[str, List[Action]],
        totals: Dict[str, int],
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
        user_name = link.get("user_name")
        if user_name is None:
            user_name = "__no_user__"
        user_id = user_store.get_id_from_name(user_name)
        try:
            user = user_store.get_user_by_id(user_id)
        except KeyError:
            user = User(user_name, {
                "can_create_topic": False,
            })
            user_store.store_user(user)
            totals["users"] += 1
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
            totals[vtype] += casts
        totals["links"] += 1
        return None
    if is_message_action(action):
        assert action["message"] is not None
        message = action["message"]
        text = message["text"]
        is_topic = False
        if text.startswith("r/") and text[2:] in roots:
            tmp = Message(msg=f"t/{text[2:]}")
            if tmp.is_topic():
                text = tmp.get_text()
                is_topic = True
        msg = Message(msg=text)
        if is_topic and msg not in message_store.get_topics():
            message_store.add_topic(msg)
            print(f"adding topic: {msg.get_text()}")
            totals["topics"] += 1
        try:
            msg = message_store.read_message(msg.get_hash())
            mhash = msg.get_hash()
        except KeyError:
            mhash = message_store.write_message(msg)
            totals["messages"] += 1
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
        roots: Set[str],
        hash_lookup: Dict[str, MHash],
        lookup_buffer: DefaultDict[str, List[Action]],
        topic_counts: DefaultDict[str, int]) -> pd.Timestamp:
    totals: Dict[str, int] = collections.defaultdict(lambda: 0)

    def print_progress() -> None:
        if totals:
            print("---")
            for vtype, casts in totals.items():
                print(f"{vtype}: {casts}")
            # totals.clear()

    for action in actions:
        if totals.get("links", 0) >= 10000:
            print_progress()
        ref = interpret_action(
            action,
            message_store=message_store,
            link_store=link_store,
            user_store=user_store,
            now=now,
            roots=roots,
            hash_lookup=hash_lookup,
            lookup_buffer=lookup_buffer,
            totals=totals)
        if ref is not None:
            ref_id, is_topic = ref
            if is_topic:
                prev_counts = max(topic_counts.values(), default=0)
                topic_counts[ref_id] += 1
                # NOTE: time hack!
                if topic_counts[ref_id] // 100 > prev_counts // 100:
                    now += pd.Timedelta("1d")
                    print(f"advance date to {now}")
            lb_actions = lookup_buffer.pop(ref_id, None)
            if lb_actions is not None and lb_actions:
                print(f"processing delayed actions ({len(lb_actions)})")
                now = process_actions(
                    lb_actions,
                    message_store=message_store,
                    link_store=link_store,
                    user_store=user_store,
                    now=now,
                    roots=roots,
                    hash_lookup=hash_lookup,
                    lookup_buffer=lookup_buffer,
                    topic_counts=topic_counts)
    print_progress()
    return now


def process_action_file(
        fname: str,
        *,
        message_store: MessageStore,
        link_store: LinkStore,
        user_store: UserStore,
        now: pd.Timestamp,
        roots: Set[str]) -> None:
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
        roots=roots,
        hash_lookup=hash_lookup,
        lookup_buffer=lookup_buffer,
        topic_counts=topic_counts)
