import time
from typing import Dict, List, Optional, Set

import pandas as pd

from example.loader import process_action_file
from system.links.link import Link, VoteType, VT_UP
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
        parent: MHash,
        already: Set[MHash]) -> None:
    already.add(parent)
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
        child = link.get_child()
        if child not in already:
            print_links(link_store, user_store, scorer, now, child, already)


def test_loader() -> None:
    message_store = get_message_store("ram")
    link_store = get_link_store("redis")
    user_store = get_user_store("ram")
    now = pd.Timestamp("2022-08-22", tz="UTC")

    msgs_raw = [
        "t/news",
        "msg 1 > root",
        "msg 2 > msg 1",
        "msg 3 > msg 1",
        "msg 4 > root",
        "msg 5 > msg 4",
        "msg 6 > msg 5",
        "msg 7 > root",
        "msg 7 > msg 5",
        "msg 8 > msg 7",
        "msg 8 > msg 7",
    ]
    msgs = [MHash.from_message(text) for text in msgs_raw]
    msgs_lookup = {MHash.from_message(text): text for text in msgs_raw}

    def print_hook(mhash: MHash) -> str:
        try:
            index = msgs.index(mhash)
        except ValueError:
            index = -1
        return (
            f"m{mhash.to_parseable()[:4]} "
            f"({index}: '{msgs_lookup.get(mhash, 'unknown')}')")

    set_mhash_print_hook(print_hook)

    reference_time = time.monotonic()
    now = process_action_file(
        "test/data/loader.jsonl",
        message_store=message_store,
        link_store=link_store,
        user_store=user_store,
        now=now,
        reference_time=reference_time,
        roots={"news"})

    settle_elapsed = link_store.settle_all()
    print(f"settle: {settle_elapsed}s")

    scorer_new = get_scorer("new")
    root = list(message_store.get_topics())[0].get_hash()

    print_links(link_store, user_store, scorer_new, now, root, set())

    def match_link(
            parent: MHash,
            child: MHash,
            uname: Optional[str],
            votes: Dict[VoteType, int],
            voters: Optional[Dict[VoteType, Set[str]]] = None) -> None:
        match_cfg(link_store.get_link(parent, child), uname, votes, voters)

    def match_cfg(
            link: Link,
            uname: Optional[str],
            votes: Dict[VoteType, int],
            voters: Optional[Dict[VoteType, Set[str]]] = None) -> None:
        user = link.get_user(user_store)
        if user is None:
            assert uname is None
        else:
            assert user.get_name() == uname
        vtypes: List[VoteType] = ["up", "down", "honor"]
        for vtype in vtypes:
            assert int(link.get_votes(vtype).get_total_votes()) == \
                votes.get(vtype, 0)
        if voters is not None:
            for (vtype, vusers) in voters.items():
                lusers = link.get_votes(vtype).get_voters(user_store)
                for vuser in vusers:
                    vuser_id = user_store.get_id_from_name(vuser)
                    assert user_store.get_user_by_id(vuser_id) in lusers

    root_links = list(link_store.get_children(
        root,
        scorer=scorer_new,
        now=now,
        offset=0,
        limit=10))
    assert len(root_links) == 3
    assert root_links[0].get_child() == msgs[7]
    assert root_links[1].get_child() == msgs[4]
    assert root_links[2].get_child() == msgs[1]

    match_cfg(root_links[0], "u/aaa", {"up": 2, "down": 1, "honor": 1})
    match_cfg(root_links[1], "u/ddd", {"up": 1, "down": 122})
    match_cfg(root_links[2], "u/aaa", {"up": 3397})

    match_link(
        msgs[7],
        msgs[9],
        "u/aaa",
        {"up": 12},
        {
            VT_UP: {
                "u/aaa", "u/bbb", "u/ccc", "u/ddd", "u/eee", "u/hhh", "u/iii"
            },
        })
    match_link(msgs[7], msgs[5], "u/iii", {"down": 4, "up": 4})
    match_link(msgs[5], msgs[6], "u/aaa", {"up": 1}, {VT_UP: {"u/aaa"}})
    match_link(msgs[4], msgs[5], "u/ddd", {"down": 46, "up": 46, "honor": 5})
    match_link(msgs[1], msgs[2], "u/bbb", {"down": 5, "up": 146})
    match_link(msgs[1], msgs[3], "u/ccc", {"up": 211})
    match_link(msgs[6], msgs[2], None, {})

    assert False
