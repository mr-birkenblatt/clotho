
import collections
import time
from typing import Iterable

import pandas as pd

from example.action import (
    Action,
    is_link_action,
    is_message_action,
    parse_action,
)
from misc.io import open_read
from misc.util import from_timestamp, read_jsonl
from system.links.link import parse_vote_type, VT_DOWN, VT_UP
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
        roots: set[str],
        hash_lookup: dict[str, MHash],
        lookup_buffer: collections.defaultdict[str, list[Action]],
        totals: dict[str, int],
        user_pool: list[User],
        synth_pool: list[User],
        ) -> tuple[str, bool] | None:
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
            user_pool.append(user)
            totals["users"] += 1
        created_ts = from_timestamp(link["created_utc"])
        any_new = False
        if "up" not in link["votes"]:
            link["votes"]["up"] = 0
        for vname, vcount in link["votes"].items():
            vtype = parse_vote_type(TYPE_CONVERTER.get(vname, "honor"))
            prev_votes = cur_link.get_votes(vtype)
            total_votes = int(prev_votes.get_total_votes())
            casts = vcount - total_votes
            first_users: list[User] = []
            prev_users = prev_votes.get_voters(user_store)
            if vtype == VT_UP:
                down_votes = link["votes"].get("down", 0)
                if down_votes > 0:
                    casts = 1 + vcount - total_votes
                first_users = [] if user in prev_users else [user]
            elif vtype == VT_DOWN:
                if casts > 0:
                    casts += 1
                    prev_users.add(user)
            if casts <= 0:
                continue
            any_new = True
            pigeon_count = casts + len(first_users) + len(prev_users)
            if casts > len(first_users):
                cur_user_pool = set(
                    user_pool[:pigeon_count]) - set(first_users) - prev_users
            else:
                cur_user_pool = set()
            if casts > len(cur_user_pool) + len(first_users):
                cur_synth_pool = set(
                    synth_pool[:pigeon_count]) - set(first_users) - prev_users
            else:
                cur_synth_pool = set()
            for _ in range(casts):
                if first_users:
                    vote_user = first_users.pop(0)
                elif cur_user_pool:
                    vote_user = cur_user_pool.pop()
                elif cur_synth_pool:
                    vote_user = cur_synth_pool.pop()
                else:
                    vote_user = User(f"s/{totals.get('users_synth', 0)}", {
                        "can_create_topic": False,
                    })
                    user_store.store_user(vote_user)
                    synth_pool.append(vote_user)
                    totals["users_synth"] += 1
                cur_link.add_vote(
                    user_store,
                    vtype,
                    vote_user,
                    now if total_votes > 0 else created_ts)
            totals[vtype] += casts
        if any_new:
            totals["new_links"] += 1
        totals["links"] += 1
        return None
    if is_message_action(action):
        assert action["message"] is not None
        message = action["message"]
        text = message["text"].strip().replace("\r", "")
        if not text:
            text = "[missing]"
        is_topic = False
        if text.startswith("r/") and text[2:].lower() in roots:
            tmp = Message(msg=f"t/{text[2:].lower()}")
            if tmp.is_topic():
                text = tmp.get_text()
                is_topic = True
        msg = Message(msg=text)
        if is_topic:
            topics = list(message_store.get_topics())
            if msg not in topics:
                message_store.add_topic(msg)
                print(f"adding topic: {msg.get_text()}")
                totals["new_topics"] += 1
            totals["topics"] = len(topics)
        try:
            msg = message_store.read_message(msg.get_hash())
            mhash = msg.get_hash()
        except KeyError:
            mhash = message_store.write_message(msg)
            totals["new_messages"] += 1
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
        reference_time: float,
        roots: set[str],
        hash_lookup: dict[str, MHash],
        lookup_buffer: collections.defaultdict[str, list[Action]],
        topic_counts: collections.defaultdict[str, int],
        totals: dict[str, int],
        user_pool: list[User],
        synth_pool: list[User],
        counter: int) -> tuple[int, pd.Timestamp]:

    def print_progress(epoch: int, total: int) -> None:
        if totals:
            print(f"---{epoch}--- ({total})")
            for key, count in sorted(totals.items()):
                print(f"{key}: {count}")
            print(f"elapsed: {time.monotonic() - reference_time:.2f}s")
            for key in list(totals.keys()):
                if not key.startswith("new_"):
                    continue
                totals.pop(key, None)
        settled, settle_timing = link_store.settle_all()
        if settled:
            print(f"settled {settled} variables in {settle_timing:.2f}s")

    try:
        for action in actions:
            counter += 1
            if counter % 10000 == 0:
                print_progress(counter // 10000, counter)
            ref = interpret_action(
                action,
                message_store=message_store,
                link_store=link_store,
                user_store=user_store,
                now=now,
                roots=roots,
                hash_lookup=hash_lookup,
                lookup_buffer=lookup_buffer,
                totals=totals,
                user_pool=user_pool,
                synth_pool=synth_pool)
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
                    counter, now = process_actions(
                        lb_actions,
                        message_store=message_store,
                        link_store=link_store,
                        user_store=user_store,
                        now=now,
                        reference_time=reference_time,
                        roots=roots,
                        hash_lookup=hash_lookup,
                        lookup_buffer=lookup_buffer,
                        topic_counts=topic_counts,
                        totals=totals,
                        user_pool=user_pool,
                        synth_pool=synth_pool,
                        counter=counter)
    finally:
        print_progress(counter // 10000, counter)
    return (counter, now)


def process_action_file(
        fname: str,
        *,
        message_store: MessageStore,
        link_store: LinkStore,
        user_store: UserStore,
        now: pd.Timestamp,
        reference_time: float,
        roots: set[str]) -> tuple[int, pd.Timestamp]:
    hash_lookup: dict[str, MHash] = {}
    lookup_buffer: collections.defaultdict[str, list[Action]] = \
        collections.defaultdict(list)
    topic_counts: collections.defaultdict[str, int] = \
        collections.defaultdict(lambda: 0)
    totals: dict[str, int] = collections.defaultdict(lambda: 0)
    user_pool: list[User] = list(user_store.get_all_users())
    if user_pool:
        print(f"loaded {len(user_pool)} users")
    synth_pool: list[User] = []
    counter = 0
    return process_actions(
        actions_from_file(fname),
        message_store=message_store,
        link_store=link_store,
        user_store=user_store,
        now=now,
        reference_time=reference_time,
        roots=roots,
        hash_lookup=hash_lookup,
        lookup_buffer=lookup_buffer,
        topic_counts=topic_counts,
        totals=totals,
        user_pool=user_pool,
        synth_pool=synth_pool,
        counter=counter)
