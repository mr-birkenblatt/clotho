from typing import Iterable, TypedDict

import numpy as np
import pandas as pd

from misc.util import now_ts, sigmoid
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

    def get_valid_links_from_messages(
            self,
            messages: list[MHash],
            scorer: Scorer,
            now: pd.Timestamp) -> list[Link]:
        rng = self._rng
        links = self._links

        def get_link(msg: MHash) -> Link:
            pcount = max(links.get_all_parents_count(msg, now), 1)
            res = list(links.get_parents(
                msg,
                scorer=scorer,
                now=now,
                offset=int(rng.integers(0, pcount)),
                limit=1))
            if not res:
                res = [links.get_link(msg, self.get_random_messages(1)[0])]
            return res[0]

        return [
            get_link(msg)
            for msg in messages
        ]

    def get_random_links_from_messages(
            self, parents: list[MHash], children: list[MHash]) -> list[Link]:
        assert len(parents) == len(children)
        links = self._links
        return [
            links.get_link(parent, child)
            for (parent, child) in zip(parents, children)
        ]

    def get_pc_flip_link(self, link: Link) -> Link:
        links = self._links
        return links.get_link(link.get_child(), link.get_parent())

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

    def get_links_from_paths(
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

    def get_valid_random_links(
            self,
            half_count: int,
            scorer: Scorer | None = None,
            now: pd.Timestamp | None = None) -> list[Link]:
        if scorer is None:
            scorer = get_scorer("best")
        if now is None:
            now = now_ts()
        messages = self.get_random_messages(half_count)
        rlinks = self.get_valid_links_from_messages(messages, scorer, now)
        paths = self.get_random_paths(half_count)
        plinks = self.get_links_from_paths(paths, scorer, now)
        return [link for pair in zip(rlinks, plinks) for link in pair]

    def get_truly_random_links(self, count: int) -> list[Link]:
        parents = self.get_random_messages(count)
        children = self.get_random_messages(count)
        return self.get_random_links_from_messages(parents, children)

    def get_flip_lrs(self, count: int) -> list[bool]:
        rng = self._rng
        return (rng.random(size=count) < 0.5).tolist()

    def get_flip_pcs(self, count: int) -> list[bool]:
        rng = self._rng
        return (rng.random(size=count) < 0.01).tolist()

    def short_info(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).to_debug()

    def long_info(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).to_debug(False)

    def vote_score(self, link: Link) -> float:
        vhonor = link.get_votes("honor").get_total_votes()
        vup = link.get_votes("up").get_total_votes()
        vdown = link.get_votes("down").get_total_votes()
        return abs(vup - HONOR_MUL * vhonor - vdown)

    def get_text(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).single_line_text()


BatchRow = TypedDict('BatchRow', {
    "parent_left": str,
    "child_left": str,
    "parent_right": str,
    "child_right": str,
    "sway_left": float,
    "sway_right": float,
    "correct_is_right": bool,
})
COLUMNS = [
    "parent_left",
    "child_left",
    "parent_right",
    "child_right",
    "sway_left",
    "sway_right",
    "correct_is_right",
]


class TrainTestGenerator:
    def __init__(
            self,
            *,
            train: DataGenerator,
            train_validation: DataGenerator,
            test: DataGenerator,
            batch_size: int,
            epoch_batches: int,
            compute_batch_size: int | None = None,
            reset_train: bool = False,
            scorer: Scorer | None = None,
            now: pd.Timestamp | None = None,
            ) -> None:
        assert train is not train_validation
        assert train_validation is not test
        assert test is not train
        self._train = train
        self._train_validation = train_validation
        self._test = test

        self._batch_size = batch_size
        cbs = batch_size if compute_batch_size is None else compute_batch_size
        assert cbs > 1
        self._half_compute_batch_size = (cbs + 1) // 2
        self._epoch_batches = epoch_batches
        self._reset_train = reset_train

        self._scorer = get_scorer("best") if scorer is None else scorer
        self._now = now_ts() if now is None else now

        self._train_buff: list[BatchRow] = []
        self._train_validation_buff: list[BatchRow] = []
        self._test_buff: list[BatchRow] = []
        self._cur_train_batch = 0
        self._cur_train_validation_batch = 0
        self._cur_test_batch = 0
        self._cur_epoch = 0

    def advance_epoch(self) -> None:
        epoch_batches = self._epoch_batches
        if (self._cur_train_batch != epoch_batches
                or self._cur_train_validation_batch != epoch_batches
                or self._cur_test_batch != epoch_batches):
            raise ValueError(
                "epoch not exhausted! "
                f"train: {self._cur_train_batch} "
                f"train validation: {self._cur_train_validation_batch} "
                f"test: {self._cur_test_batch} "
                f"batches per epoch: {epoch_batches}")
        if self._reset_train:
            self._train.reset()
        self._train_validation.reset()
        self._test.reset()
        self._train_buff = []
        self._train_validation_buff = []
        self._test_buff = []
        self._cur_train_batch = 0
        self._cur_train_validation_batch = 0
        self._cur_test_batch = 0
        self._cur_epoch += 1

    def get_epoch(self) -> int:
        return self._cur_epoch

    def _compute_row(
            self,
            data: DataGenerator,
            left: Link,
            right: Link,
            flip_lr: bool,
            flip_pc: bool) -> BatchRow:
        if flip_pc:
            left = data.get_pc_flip_link(right)
        if flip_lr:
            left, right = right, left
        score_left = data.vote_score(left)
        score_right = data.vote_score(right)
        sway_right = float(sigmoid(score_right - score_left))
        return {
            "parent_left": data.get_text(left.get_parent()),
            "child_left": data.get_text(left.get_child()),
            "parent_right": data.get_text(right.get_parent()),
            "child_right": data.get_text(right.get_child()),
            "sway_left": 1.0 - sway_right,
            "sway_right": sway_right,
            "correct_is_right": score_right > score_left,
        }

    def _compute_batch_for(
            self, data: DataGenerator, buff: list[BatchRow]) -> None:
        cbs = self._half_compute_batch_size * 2
        randos = data.get_truly_random_links(cbs)
        valids = data.get_valid_random_links(
            self._half_compute_batch_size, self._scorer, self._now)
        flip_lrs = data.get_flip_lrs(cbs)
        flip_pcs = data.get_flip_pcs(cbs)
        assert len(valids) == len(randos)
        assert len(valids) == len(flip_lrs)
        assert len(valids) == len(flip_pcs)
        for row in zip(randos, valids, flip_lrs, flip_pcs):
            rando, valid, flip_lr, flip_pc = row
            buff.append(
                self._compute_row(data, rando, valid, flip_lr, flip_pc))

    def _get_batch_for(
            self,
            data: DataGenerator,
            buff: list[BatchRow]) -> tuple[list[BatchRow], list[BatchRow]]:
        batch_size = self._batch_size
        while len(buff) < batch_size:
            self._compute_batch_for(data, buff)
        return buff[:batch_size], buff[batch_size:]

    def next_train_batch(self) -> list[BatchRow] | None:
        if self._cur_train_batch >= self._epoch_batches:
            return None
        res, self._train_buff = self._get_batch_for(
            self._train, self._train_buff)
        self._cur_train_batch += 1
        return res

    def next_train_validation_batch(self) -> list[BatchRow] | None:
        if self._cur_train_validation_batch >= self._epoch_batches:
            return None
        res, self._train_validation_buff = self._get_batch_for(
            self._train_validation, self._train_validation_buff)
        self._cur_train_validation_batch += 1
        return res

    def next_test_batch(self) -> list[BatchRow] | None:
        if self._cur_test_batch >= self._epoch_batches:
            return None
        res, self._test_buff = self._get_batch_for(self._test, self._test_buff)
        self._cur_test_batch += 1
        return res

    def train_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_train_batch >= self._epoch_batches:
            raise ValueError("train batches already exhausted!")
        while True:
            res = self.next_train_batch()
            if res is None:
                return
            yield res

    def train_validation_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_train_validation_batch >= self._epoch_batches:
            raise ValueError("train validation batches already exhausted!")
        while True:
            res = self.next_train_validation_batch()
            if res is None:
                return
            yield res

    def test_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_test_batch >= self._epoch_batches:
            raise ValueError("test batches already exhausted!")
        while True:
            res = self.next_test_batch()
            if res is None:
                return
            yield res

    def train_dfs(self) -> Iterable[pd.DataFrame]:
        yield from (
            pd.DataFrame(val, columns=COLUMNS)
            for val in self.train_batches()
        )

    def train_validation_dfs(self) -> Iterable[pd.DataFrame]:
        yield from (
            pd.DataFrame(val, columns=COLUMNS)
            for val in self.train_validation_batches()
        )

    def test_dfs(self) -> Iterable[pd.DataFrame]:
        yield from (
            pd.DataFrame(val, columns=COLUMNS)
            for val in self.test_batches()
        )


def create_train_test(
        *,
        train_ns: Namespace,
        train_validation_ns: Namespace,
        test_ns: Namespace,
        batch_size: int,
        epoch_batches: int,
        train_seed: int = 42,
        train_validation_seed: int = 37,
        test_seed: int = 69,
        compute_batch_size: int | None = None,
        reset_train: bool = False,
        scorer: Scorer | None = None,
        now: pd.Timestamp | None = None) -> TrainTestGenerator:
    return TrainTestGenerator(
        train=DataGenerator(train_ns, train_seed),
        train_validation=DataGenerator(
            train_validation_ns, train_validation_seed),
        test=DataGenerator(test_ns, test_seed),
        batch_size=batch_size,
        epoch_batches=epoch_batches,
        compute_batch_size=compute_batch_size,
        reset_train=reset_train,
        scorer=scorer,
        now=now)
