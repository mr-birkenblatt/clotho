import numpy as np
import pandas as pd

from misc.util import now_ts
from system.links.link import Link
from system.links.scorer import get_scorer, Scorer
from system.links.store import get_link_store
from system.msgs.message import MHash
from system.msgs.store import get_message_store
from system.namespace.namespace import Namespace


SEED_MUL = 17
HONOR_MUL = 10.0


class DataGenerator:
    def __init__(self, namespace: Namespace, seed: int) -> None:
        self._msgs = get_message_store(namespace)
        self._links = get_link_store(namespace)
        self._seed = seed
        self._prob_next = 0.71
        self._prob_down = 0.87
        self._rng = np.random.default_rng(abs(seed))
        self._rix = 0

    def reset(self) -> None:
        seed = self._seed
        self._rng = np.random.default_rng(abs(seed))
        self._rix = 0

    def get_random_messages(self, count: int) -> list[MHash]:
        seed = self._seed
        rix = self._rix
        self._rix += count

        def get_rng(cur_ix: int) -> np.random.Generator:
            return np.random.default_rng(abs(seed + SEED_MUL * cur_ix))

        return self._msgs.generate_random_messages(
            get_rng, rix, count)

    def get_valid_link_from_messages(
            self,
            messages: list[MHash],
            scorer: Scorer,
            now: pd.Timestamp) -> list[Link]:
        links = self._links

        def get_link(msg: MHash) -> Link:
            res = list(links.get_children(
                msg, scorer=scorer, now=now, offset=0, limit=1))
            if not res:
                res = [links.get_link(msg, self.get_random_messages(1)[0])]
            return res[0]

        return [
            get_link(msg)
            for msg in messages
        ]

    def get_random_link_from_messages(
            self, parents: list[MHash], children: list[MHash]) -> list[Link]:
        assert len(parents) == len(children)
        links = self._links
        return [
            links.get_link(parent, child)
            for (parent, child) in zip(parents, children)
        ]

    def get_random_paths(self, count: int) -> list[list[int]]:
        rng = self._rng
        prob_next = self._prob_next
        prob_down = self._prob_down

        def get_path() -> list[int]:
            cur = []
            cur_ix = 0
            while True:
                if rng.random() < prob_next:
                    cur_ix += 1
                    continue
                if rng.random() < prob_down:
                    cur.append(cur_ix)
                    cur_ix = 0
                    continue
                return cur

        return [get_path() for _ in range(count)]

    def get_link_from_paths(
            self,
            paths: list[list[int]],
            scorer: Scorer,
            now: pd.Timestamp) -> list[Link]:
        rng = self._rng
        msgs = self._msgs
        links = self._links

        def get_link(path: list[int]) -> Link:
            topic_count = max(msgs.get_topics_count(), 1)
            if len(path) < 1:
                path.append(int(rng.integers(0, topic_count)))
            parent, = msgs.get_topics(path[0] % topic_count, 1)
            phash = parent.get_hash()
            if len(path) < 2:
                init_ccount = max(links.get_all_children_count(phash, now), 1)
                path.append(int(rng.integers(0, init_ccount)))
            res = None
            for cur in path[1:]:
                count = links.get_all_children_count(phash, now)
                if count == 0:
                    break
                mres = list(links.get_children(
                    phash,
                    scorer=scorer,
                    now=now,
                    offset=cur % count,
                    limit=1))
                if not mres:
                    break
                res, = mres
                phash = res.get_child()
            if res is None:
                mhash, = self.get_random_messages(1)
                res = links.get_link(phash, mhash)
            return res

        return [get_link(path) for path in paths]

    def get_random_links(
            self,
            half_count: int,
            scorer: Scorer | None = None,
            now: pd.Timestamp | None = None) -> list[Link]:
        if scorer is None:
            scorer = get_scorer("best")
        if now is None:
            now = now_ts()
        messages = self.get_random_messages(half_count)
        rlinks = self.get_valid_link_from_messages(messages, scorer, now)
        paths = self.get_random_paths(half_count)
        plinks = self.get_link_from_paths(paths, scorer, now)
        return [link for pair in zip(rlinks, plinks) for link in pair]

    def short_info(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).to_debug()

    def vote_score(self, link: Link) -> float:
        vhonor = link.get_votes("honor").get_total_votes()
        vup = link.get_votes("up").get_total_votes()
        vdown = link.get_votes("down").get_total_votes()
        return abs(vup - HONOR_MUL * vhonor - vdown)
