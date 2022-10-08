import time
from typing import List

import pandas as pd

from example.loader import process_action_file
from system.links.link import Link
from system.links.scorer import get_scorer, Scorer
from system.links.store import get_link_store, LinkStore
from system.msgs.message import MHash, set_mhash_print_hook
from system.msgs.store import get_message_store
from system.users.store import get_user_store, UserStore


def get_all_children(
        link_store: LinkStore,
        scorer: Scorer,
        now: pd.Timestamp,
        parent: MHash) -> List[Link]:
    offset = 0
    limit = 10
    res: List[Link] = []
    while True:
        cur = list(link_store.get_children(
            parent,
            scorer=scorer,
            now=now,
            offset=offset,
            limit=limit))
        if not cur:
            break
        res.extend(cur)
        offset += limit
    return res


def print_links(
        link_store: LinkStore,
        user_store: UserStore,
        scorer: Scorer,
        now: pd.Timestamp,
        parent: MHash) -> None:
    links = get_all_children(link_store, scorer, now, parent)
    for link in links:
        votes = ", ".join((
            f"{vtype}={int(link.get_votes(vtype).get_total_votes())}"
            for vtype in link.get_vote_types()
            if int(link.get_votes(vtype).get_total_votes()) != 0
        ))
        user = link.get_user(user_store)
        user_str = None if user is None else user.get_name()
        print(f"{link} ({user_str}; {votes})")
    for link in links:
        print_links(link_store, user_store, scorer, now, link.get_child())


def test_loader() -> None:
    message_store = get_message_store("ram")
    link_store = get_link_store("redis")
    user_store = get_user_store("ram")
    now = pd.Timestamp("2022-08-22", tz="UTC")

    set_mhash_print_hook(lambda mhash: f"m{mhash.to_parseable()[:4]}")

    reference_time = time.monotonic()
    now = process_action_file(
        "test/data/loader.jsonl",
        message_store=message_store,
        link_store=link_store,
        user_store=user_store,
        now=now,
        reference_time=reference_time,
        roots={"news"})

    print(f"settle: {link_store.settle_all()}s")

    scorer_new = get_scorer("new")
    root = list(message_store.get_topics())[0].get_hash()

    print_links(link_store, user_store, scorer_new, now, root)

    root_links = list(link_store.get_children(
        root,
        scorer=scorer_new,
        now=now,
        offset=0,
        limit=10))
    assert len(root_links) == 3
    assert root_links[0].get_child() == MHash.from_message("msg 7 > root")
    assert root_links[1].get_child() == MHash.from_message("msg 4 > root")
    assert root_links[2].get_child() == MHash.from_message("msg 1 > root")

    assert False
