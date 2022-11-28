
import time
import uuid
from typing import Iterable

from misc.util import from_timestamp, now_ts, to_timestamp
from system.links.link import Link, VoteType, VT_DOWN, VT_UP
from system.links.redisstore import set_delay_multiplier
from system.links.scorer import get_scorer, ScorerName
from system.links.store import get_link_store
from system.msgs.message import MHash, set_mhash_print_hook
from system.users.store import get_user_store
from system.users.user import User


def get_random_mhash() -> MHash:
    return MHash.from_message(uuid.uuid4().hex)


PLAN: list[tuple[int, int, int, VoteType]] = [
    # parent, child, user, vote_type
    (0, 1, 0, VT_UP),
    (0, 3, 0, VT_UP),
    (0, 2, 0, VT_UP),
    (4, 5, 1, VT_UP),
    (4, 6, 1, VT_UP),
    (4, 7, 1, VT_UP),
    (3, 1, 2, VT_UP),
    (3, 5, 2, VT_UP),
    (3, 7, 2, VT_UP),
    (7, 8, 3, VT_UP),
    (7, 9, 3, VT_UP),
    (0, 1, 1, VT_UP),
    (0, 1, 2, VT_UP),
    (0, 2, 1, VT_UP),
    (0, 1, 4, VT_DOWN),
    (0, 1, 4, VT_DOWN),  # intentional duplicate
    (0, 3, 4, VT_DOWN),
    (0, 3, 0, VT_DOWN),
    (0, 3, 1, VT_DOWN),
    (0, 3, 2, VT_DOWN),
]


DMUL = 0.05


def test_scenario() -> None:
    msgs = [get_random_mhash() for _ in range(10)]
    mlookup = {mhash: f"msgs[{ix}]" for (ix, mhash) in enumerate(msgs)}
    set_mhash_print_hook(
        lambda mhash: mlookup.get(mhash, mhash.to_parseable()))

    users = [
        User(f"u{uid}", {
            "can_create_topic": False,
        })
        for uid in range(5)
    ]
    user_store = get_user_store("ram")
    for user in users:
        user_store.store_user(user)

    dmul = DMUL
    set_delay_multiplier(dmul)
    store = get_link_store("redis")

    def get_link(parent: int, child: int) -> Link:
        return store.get_link(msgs[parent], msgs[child])

    first_s = float(int(to_timestamp(now_ts()) - len(PLAN) * 10.0))
    now_s = first_s
    for action in PLAN:
        a_p, a_c, a_u, avt = action
        get_link(a_p, a_c).add_vote(
            user_store, avt, users[a_u], from_timestamp(now_s))
        now_s += 10.0

    now = from_timestamp(now_s)

    assert int(get_link(0, 1).get_votes(VT_UP).get_total_votes()) == 3
    assert int(get_link(0, 1).get_votes(VT_DOWN).get_total_votes()) == 1

    time.sleep(4.0 * dmul)  # update tier 1
    # (all parents, all children)

    def get_children(links: Iterable[Link]) -> list[MHash]:
        return [link.get_child() for link in links]

    def get_parents(links: Iterable[Link]) -> list[MHash]:
        return [link.get_parent() for link in links]

    assert set(get_children(store.get_all_children(msgs[0]))) == {
        msgs[1], msgs[2], msgs[3]}
    assert set(get_parents(store.get_all_children(msgs[0]))) == {msgs[0]}

    assert set(get_children(store.get_all_children(msgs[4]))) == {
        msgs[5], msgs[6], msgs[7]}

    assert set(get_children(store.get_all_children(msgs[3]))) == {
        msgs[1], msgs[5], msgs[7]}

    assert len(get_children(store.get_all_children(msgs[2]))) == 0

    assert set(get_parents(store.get_all_parents(msgs[7]))) == {
        msgs[3], msgs[4]}
    assert set(get_children(store.get_all_parents(msgs[7]))) == {msgs[7]}

    assert set(get_parents(store.get_all_parents(msgs[1]))) == {
        msgs[0], msgs[3]}

    assert set(get_parents(store.get_all_parents(msgs[5]))) == {
        msgs[3], msgs[4]}

    assert set(get_parents(store.get_all_parents(msgs[2]))) == {msgs[0]}
    assert set(get_parents(store.get_all_parents(msgs[9]))) == {msgs[7]}
    assert len(get_parents(store.get_all_parents(msgs[4]))) == 0

    time.sleep(5.0 * dmul)  # update tier 2
    # (sorted parents, sorted children, first user, user list)

    def get_sorted(
            parent: int,
            scorer_name: ScorerName,
            *,
            is_children: bool,
            full: bool) -> list[MHash]:
        scorer = get_scorer(scorer_name)
        sfn = store.get_children if is_children else store.get_parents
        ofn = get_children if is_children else get_parents
        if full:
            return ofn(sfn(
                msgs[parent],
                scorer=scorer,
                now=now,
                offset=0,
                limit=10))
        res = []
        off = 0
        while True:
            cur = ofn(sfn(
                msgs[parent],
                scorer=scorer,
                now=now,
                offset=off,
                limit=2))
            if not cur:
                break
            res.extend(cur)
            off += 2
        return res

    assert get_sorted(4, "new", is_children=True, full=True) == [
        msgs[7], msgs[6], msgs[5]]
    assert get_sorted(4, "new", is_children=True, full=False) == [
        msgs[7], msgs[6], msgs[5]]

    assert get_sorted(0, "top", is_children=True, full=True) == [
        msgs[1], msgs[2], msgs[3]]
    assert get_sorted(0, "top", is_children=True, full=False) == [
        msgs[1], msgs[2], msgs[3]]
    assert get_sorted(0, "best", is_children=True, full=True) == [
        msgs[1], msgs[3], msgs[2]]
    assert get_sorted(0, "new", is_children=True, full=True) == [
        msgs[2], msgs[3], msgs[1]]

    assert get_sorted(7, "new", is_children=False, full=True) == [
        msgs[3], msgs[4]]
    assert get_sorted(7, "new", is_children=False, full=False) == [
        msgs[3], msgs[4]]

    assert len(get_sorted(0, "top", is_children=False, full=False)) == 0

    resp = get_link(0, 1).get_response(user_store, now)
    assert resp["parent"] == msgs[0].to_parseable()
    assert resp["child"] == msgs[1].to_parseable()
    assert resp["user"] == users[0].get_id()
    assert int(resp["first"]) == int(first_s)
    rvotes = resp["votes"]
    assert rvotes.keys() == {VT_UP, VT_DOWN}
    assert int(rvotes[VT_UP]["count"]) == 3
    assert int(rvotes[VT_DOWN]["count"]) == 1

    resp = get_link(0, 3).get_response(user_store, now)
    assert resp["parent"] == msgs[0].to_parseable()
    assert resp["child"] == msgs[3].to_parseable()
    assert resp["user"] == users[0].get_id()
    assert int(resp["first"]) == int(first_s + 10.0)
    rvotes = resp["votes"]
    assert rvotes.keys() == {VT_UP, VT_DOWN}
    assert int(rvotes[VT_UP]["count"]) == 1
    assert int(rvotes[VT_DOWN]["count"]) == 4

    assert get_link(0, 3).get_votes(VT_DOWN).get_voters(user_store) == {
        users[0], users[1], users[2], users[4]}

    assert set(store.get_all_user_links(users[0])) == \
        set(store.get_all_children(msgs[0]))
    assert set(store.get_all_user_links(users[2])) == \
        set(store.get_all_children(msgs[3]))

    time.sleep(4.0 * dmul)  # update tier 3
    # (sorted user list)

    def get_sorted_user(
            user: int,
            scorer_name: ScorerName,
            *,
            is_children: bool,
            full: bool) -> list[MHash]:
        scorer = get_scorer(scorer_name)
        ofn = get_children if is_children else get_parents
        if full:
            return ofn(store.get_user_links(
                users[user],
                scorer=scorer,
                now=now,
                offset=0,
                limit=10))
        res = []
        off = 0
        while True:
            cur = ofn(store.get_user_links(
                users[user],
                scorer=scorer,
                now=now,
                offset=off,
                limit=2))
            if not cur:
                break
            res.extend(cur)
            off += 2
        return res

    assert get_sorted_user(2, "new", is_children=True, full=True) == [
        msgs[7], msgs[5], msgs[1]]
    assert get_sorted_user(2, "new", is_children=False, full=True) == [
        msgs[3], msgs[3], msgs[3]]
    assert get_sorted_user(2, "new", is_children=True, full=False) == [
        msgs[7], msgs[5], msgs[1]]

    assert get_sorted_user(0, "top", is_children=True, full=True) == [
        msgs[1], msgs[2], msgs[3]]
    assert get_sorted_user(0, "top", is_children=True, full=False) == [
        msgs[1], msgs[2], msgs[3]]
    assert get_sorted_user(0, "best", is_children=True, full=True) == [
        msgs[1], msgs[3], msgs[2]]
    assert get_sorted_user(0, "new", is_children=True, full=True) == [
        msgs[2], msgs[3], msgs[1]]
