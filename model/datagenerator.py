import collections
import threading
from typing import Callable, Iterable, Literal, Sequence, TypedDict, TypeVar

import numpy as np
import pandas as pd

from misc.lru import LRU
from misc.util import now_ts, sigmoid
from system.links.link import Link
from system.links.scorer import get_scorer, Scorer
from system.links.store import get_link_store
from system.msgs.message import MHash
from system.msgs.store import get_message_store
from system.namespace.namespace import Namespace


SEED_OFFSET_MUL = 37
SEED_MUL = 17
HONOR_MUL = 10.0
DOWN_MUL = 0.3


T = TypeVar('T')


RowMode = Literal["path", "valid", "random"]
RowGen = TypedDict('RowGen', {
    "mode": RowMode,
    "flip_pc": float,
})
LearningPlan = TypedDict('LearningPlan', {
    "left": RowGen | None,
    "right": RowGen,
    "min_text_length": int | None,
    "skip_weak": bool,
    "skip_topics": bool,
    "flip_lr": float,
    "weight": float,
})
EpochLearningPlan = TypedDict('EpochLearningPlan', {
    "left": RowGen | None,
    "right": RowGen,
    "first_epoch": int | None,
    "last_epoch": int | None,
    "min_text_length": int | None,
    "skip_weak": bool,
    "skip_topics": bool,
    "flip_lr": float,
    "weight": float,
})


class DataGenerator:
    def __init__(self, namespace: Namespace, seed: int) -> None:
        self._msgs = get_message_store(namespace)
        self._links = get_link_store(namespace)
        self._seed = seed
        self._prob_next = 0.71
        self._prob_down = 0.87
        self._seed_offset = 0
        self._rng = self._get_rng(0)
        self._rix = 0

    def _get_rng(self, cur_ix: int) -> np.random.Generator:
        calc = self._seed * (1 + SEED_OFFSET_MUL * self._seed_offset) + 1
        return np.random.default_rng(abs(calc * (1 + SEED_MUL * cur_ix)))

    def set_seed_offset(self, offset: int) -> None:
        self._seed_offset = offset
        self._rng = self._get_rng(0)
        self._rix = 0

    def reset(self) -> None:
        self._seed_offset = 0
        self._rng = self._get_rng(0)
        self._rix = 0

    def _get_random_messages(self, count: int) -> list[MHash]:
        rix = self._rix
        self._rix += count
        return self._msgs.generate_random_messages(
            self._get_rng, rix, count)

    def _get_valid_links_from_messages(
            self,
            messages: list[MHash],
            scorer: Scorer,
            now: pd.Timestamp) -> list[Link]:
        rng = self._rng
        links = self._links
        pool: collections.deque[MHash] = collections.deque()

        def random_message() -> MHash:
            while not pool:
                pool.extend(self._get_random_messages(100))
            return pool.popleft()

        def get_link(msg: MHash) -> Link:
            pcount = max(links.get_all_parents_count(msg, now), 1)
            res = list(links.get_parents(
                msg,
                scorer=scorer,
                now=now,
                offset=int(rng.integers(0, pcount)),
                limit=1))
            if not res:
                res = [links.get_link(msg, random_message())]
            return res[0]

        return [
            get_link(msg)
            for msg in messages
        ]

    def _get_random_links_from_messages(
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

    def _get_random_paths(self, count: int) -> list[list[int]]:
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

    def _get_links_from_paths(
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
                mhash, = self._get_random_messages(1)
                res = links.get_link(phash, mhash)
            return res

        return [get_link(path) for path in paths]

    def get_valid_random_links(
            self,
            count: int,
            scorer: Scorer,
            now: pd.Timestamp) -> list[Link]:
        messages = self._get_random_messages(count)
        return self._get_valid_links_from_messages(messages, scorer, now)

    def get_path_links(
            self,
            count: int,
            scorer: Scorer,
            now: pd.Timestamp) -> list[Link]:
        paths = self._get_random_paths(count)
        return self._get_links_from_paths(paths, scorer, now)

    def get_truly_random_links(self, count: int) -> list[Link]:
        parents = self._get_random_messages(count)
        children = self._get_random_messages(count)
        return self._get_random_links_from_messages(parents, children)

    def get_random_numbers(self, count: int) -> list[float]:
        rng = self._rng
        return rng.random(size=count).tolist()

    def get_weighted_choice(
            self, arr: list[T], weights: list[float], count: int) -> list[T]:
        rng = self._rng
        total_weight = sum(weights)
        probs = [weight / total_weight for weight in weights]
        return [
            arr[int(pix)]
            for pix in rng.choice(len(arr), count, p=probs)
        ]

    def short_info(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).to_debug()

    def long_info(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).to_debug(False)

    def vote_score(self, link: Link) -> float:
        vhonor = link.get_votes("honor").get_total_votes()
        vup = link.get_votes("up").get_total_votes()
        vdown = link.get_votes("down").get_total_votes()
        return abs(vup - HONOR_MUL * vhonor - DOWN_MUL * vdown)

    def is_weak(self, link: Link) -> bool:
        vup = link.get_votes("up").get_total_votes()
        vdown = link.get_votes("down").get_total_votes()
        if vup < 2 and vdown < 1:
            return True
        if vdown < 2 and vup < 2:
            return True
        return False

    def has_topic(self, link: Link) -> bool:
        if self.is_topic_like(link.get_parent()):
            return True
        if self.is_topic_like(link.get_child()):
            return True
        return False

    def is_topic_like(self, mhash: MHash) -> bool:
        return not self._msgs.read_message(mhash).is_valid_message()

    def get_text(self, mhash: MHash) -> str:
        return self._msgs.read_message(mhash).single_line_text()


BatchRow = TypedDict('BatchRow', {
    "gen_name": str,
    "parent_left": str,
    "child_left": str,
    "parent_right": str,
    "child_right": str,
    "sway_left": float,
    "sway_right": float,
    "correct_is_right": bool,
})
COLUMNS = [
    "gen_name",
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
            test_validation: DataGenerator,
            train_learning_plan: Sequence[EpochLearningPlan | LearningPlan],
            train_val_learning_plan: list[LearningPlan],
            test_learning_plan: list[LearningPlan],
            test_val_learning_plan: list[LearningPlan],
            batch_size: int,
            epoch_batches: int,
            train_val_size: int,
            test_size: int,
            test_val_size: int,
            compute_batch_size: int | None = None,
            scorer: Scorer | None = None,
            now: pd.Timestamp | None = None,
            ) -> None:
        assert train is not train_validation
        assert train is not test
        assert train is not test_validation
        assert train_validation is not test
        assert train_validation is not test_validation
        assert test is not test_validation
        self._train = train
        self._train_validation = train_validation
        self._test = test
        self._test_validation = test_validation
        self._train_learning_plan = train_learning_plan
        self._train_val_learning_plan = train_val_learning_plan
        self._test_learning_plan = test_learning_plan
        self._test_val_learning_plan = test_val_learning_plan

        self._batch_size = batch_size
        self._train_val_size = train_val_size
        self._test_size = test_size
        self._test_val_size = test_val_size
        self._compute_batch_size = \
            batch_size if compute_batch_size is None else compute_batch_size
        self._epoch_batches = epoch_batches

        self._scorer = get_scorer("best") if scorer is None else scorer
        self._now = now_ts() if now is None else now

        self._train_buff: collections.deque[BatchRow] = collections.deque()
        self._train_validation_buff: collections.deque[BatchRow] = \
            collections.deque()
        self._test_buff: collections.deque[BatchRow] = collections.deque()
        self._test_validation_buff: collections.deque[BatchRow] = \
            collections.deque()
        self._cur_epoch = 0

        self._train_epoch_cache: LRU[int, list[BatchRow]] = LRU(35)
        self._cur_train_cache: list[BatchRow] = []

        self._cur_train_validation_cache: list[BatchRow] = []
        self._cur_test_cache: list[BatchRow] = []
        self._cur_test_validation_cache: list[BatchRow] = []

        self._cur_train_ix = 0
        self._cur_train_validation_ix = 0
        self._cur_test_ix = 0
        self._cur_test_validation_ix = 0

        self._th_train: threading.Thread | None = None
        self._th_train_val: threading.Thread | None = None
        self._th_test: threading.Thread | None = None
        self._th_test_val: threading.Thread | None = None
        self._th_term: bool = False
        self._th_err: BaseException | None = None
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_epoch_train_validation_size(self) -> int:
        return self._train_val_size

    def get_epoch_test_size(self) -> int:
        return self._test_size

    def get_epoch_test_validation_size(self) -> int:
        return self._test_val_size

    def get_train_batches(self) -> int:
        return self._epoch_batches

    def get_epoch_train_size(self) -> int:
        return self._batch_size * self._epoch_batches

    def _th_terminate(self) -> None:
        self._th_term = True

        def done() -> bool:
            return (
                self._th_train is None
                and self._th_train_val is None
                and self._th_test is None
                and self._th_test_val is None)

        while not done():
            self._check_err()
            with self._cond:
                self._cond.wait_for(done, 1.0)

        self._check_err()
        self._th_term = False

    def _empty_train_cache(self) -> None:
        train_cache = list(self._cur_train_cache)
        if len(train_cache) == self.get_epoch_train_size():
            self._train_epoch_cache.set(self._cur_epoch, train_cache)
        self._cur_train_cache = []
        self._cur_train_ix = 0

    def reset(self) -> None:
        self._th_terminate()
        self._empty_train_cache()
        self._train.reset()
        self._train_buff.clear()
        self._cur_train_validation_ix = 0
        self._cur_test_ix = 0
        self._cur_test_validation_ix = 0
        self._cur_epoch = 0

    def advance_epoch(self) -> None:
        if (self._cur_train_ix != self.get_epoch_train_size()
                or self._cur_train_validation_ix != self._train_val_size
                or self._cur_test_ix != self._test_size):
            raise ValueError(
                "epoch not exhausted! "
                f"train: {self._cur_train_ix} "
                f"train validation: {self._cur_train_validation_ix} "
                f"test: {self._cur_test_ix} "
                f"train validation: {self._cur_test_validation_ix} "
                f"batches per epoch: {self._epoch_batches} "
                f"train val size: {self._train_val_size} "
                f"test size: {self._test_size} "
                f"test val size: {self._test_val_size}")
        self._th_terminate()
        self._empty_train_cache()
        self._set_epoch(self._cur_epoch + 1)
        self._train.set_seed_offset(self._cur_epoch)
        self._train_buff.clear()
        self._cur_train_validation_ix = 0
        self._cur_test_ix = 0
        self._cur_test_validation_ix = 0

    def get_epoch(self) -> int:
        return self._cur_epoch

    def _set_epoch(self, epoch: int) -> None:
        self._cur_epoch = epoch
        train_cache = self._train_epoch_cache.get(self._cur_epoch)
        if train_cache is not None:
            self._cur_train_cache = train_cache
        else:
            self._cur_train_cache = []

    def set_epoch(self, epoch: int) -> None:
        self.reset()
        self._set_epoch(epoch)
        self._train.set_seed_offset(self._cur_epoch)

    @staticmethod
    def _epoch_learning_plan(
            epoch: int,
            learning_plan: Sequence[EpochLearningPlan | LearningPlan],
            ) -> list[LearningPlan]:

        def after_first(
                epoch: int, plan: EpochLearningPlan | LearningPlan) -> bool:
            res: int | None = plan.get("first_epoch")  # type: ignore
            if res is None:
                return True
            return epoch >= res

        def before_last(
                epoch: int, plan: EpochLearningPlan | LearningPlan) -> bool:
            res: int | None = plan.get("last_epoch")  # type: ignore
            if res is None:
                return True
            return epoch <= res

        return [
            {
                "left": lplan["left"],
                "right": lplan["right"],
                "min_text_length": lplan["min_text_length"],
                "skip_weak": lplan["skip_weak"],
                "skip_topics": lplan["skip_topics"],
                "flip_lr": lplan["flip_lr"],
                "weight": lplan["weight"],
            }
            for lplan in learning_plan
            if after_first(epoch, lplan) and before_last(epoch, lplan)
        ]

    @staticmethod
    def _from_learning_plan(
            data: DataGenerator,
            learning_plan: list[LearningPlan],
            count: int,
            scorer: Scorer,
            now: pd.Timestamp) -> Iterable[BatchRow]:
        plan = data.get_weighted_choice(
            learning_plan,
            [lplan["weight"] for lplan in learning_plan],
            count)
        flip_lrs = data.get_random_numbers(count)
        flip_left_pc = data.get_random_numbers(count)
        flip_right_pc = data.get_random_numbers(count)
        rcounts: collections.defaultdict[RowMode, int] = \
            collections.defaultdict(lambda: 0)
        for pentry in plan:
            if pentry["left"] is not None:
                rcounts[pentry["left"]["mode"]] += 1
            rcounts[pentry["right"]["mode"]] += 1

        def gen(mode: RowMode, rcount: int) -> list[Link]:
            if mode == "random":
                return data.get_truly_random_links(rcount)
            if mode == "valid":
                return data.get_valid_random_links(rcount, scorer, now)
            if mode == "path":
                return data.get_path_links(rcount, scorer, now)
            raise ValueError(f"invalid mode: {mode}")

        links = {
            key: collections.deque(gen(key, kcount))
            for key, kcount in rcounts.items()
        }
        produced = 0
        for ix, pentry in enumerate(plan):
            right = pentry["right"]
            right_link = links[right["mode"]].popleft()
            name_right = f"{right['mode']}"
            if flip_right_pc[ix] < right["flip_pc"]:
                right_link = data.get_pc_flip_link(right_link)
                name_right = f"!{name_right}"

            left = pentry["left"]
            if left is None:
                left_link = data.get_pc_flip_link(right_link)
                name_left = "!copy"
            else:
                left_link = links[left["mode"]].popleft()
                name_left = f"{left['mode']}"
                if flip_left_pc[ix] < left["flip_pc"]:
                    left_link = data.get_pc_flip_link(left_link)
                    name_left = f"!{name_left}"

            if flip_lrs[ix] < pentry["flip_lr"]:
                left_link, right_link = right_link, left_link
                name = f"*{name_right}--{name_left}"
            else:
                name = f"{name_left}--{name_right}"

            text_pl = data.get_text(left_link.get_parent())
            text_cl = data.get_text(left_link.get_child())
            text_pr = data.get_text(right_link.get_parent())
            text_cr = data.get_text(right_link.get_child())
            mtl = pentry["min_text_length"]
            if mtl is not None:
                name = f"{name};(mtl:{mtl})"
                texts = [
                    text_pl,
                    text_cl,
                    text_pr,
                    text_cr,
                ]
                if any(len(txt) < mtl for txt in texts):
                    continue

            score_left = data.vote_score(left_link)
            score_right = data.vote_score(right_link)
            if score_left == score_right:
                continue

            if pentry["skip_weak"]:
                name = f"{name};(sw)"
                if score_right > score_left and data.is_weak(right_link):
                    continue
                if score_right < score_left and data.is_weak(left_link):
                    continue

            if pentry["skip_topics"]:
                name = f"{name};(st)"
                if data.has_topic(right_link):
                    continue
                if data.has_topic(left_link):
                    continue

            sway_right = float(sigmoid(score_right - score_left))
            yield {
                "parent_left": text_pl,
                "child_left": text_cl,
                "parent_right": text_pr,
                "child_right": text_cr,
                "sway_left": 1.0 - sway_right,
                "sway_right": sway_right,
                "correct_is_right": score_right > score_left,
                "gen_name": name,
            }
            produced += 1
        if produced == 0:
            print(
                "WARNING: current setting produced no output "
                "likely resulting in an infinite loop")

    def _compute_batch_for(
            self,
            data: DataGenerator,
            learning_plan: list[LearningPlan],
            buff: collections.deque[BatchRow]) -> None:
        for row in self._from_learning_plan(
                data,
                learning_plan,
                self._compute_batch_size,
                self._scorer,
                self._now):
            buff.append(row)
            with self._cond:
                self._cond.notify_all()

    def _th_compute_batch(
            self,
            is_alive: Callable[[], bool],
            data: DataGenerator,
            learning_plan: list[LearningPlan],
            buff: collections.deque[BatchRow]) -> None:
        while len(buff) < self._compute_batch_size * 3:
            with self._lock:
                if not is_alive():
                    return
            self._compute_batch_for(data, learning_plan, buff)

    def _get_batch_for(
            self,
            start_th: Callable[[], None],
            buff: collections.deque[BatchRow],
            out: list[BatchRow]) -> None:

        def has_rows() -> bool:
            return bool(buff)

        for _ in range(self._batch_size):
            while not has_rows():
                self._check_err()
                start_th()
                with self._cond:
                    self._cond.wait_for(has_rows, 1.0)
            out.append(buff.popleft())
        self._check_err()

    def _check_err(self) -> None:
        if self._th_err is not None:
            raise ValueError("error in compute thread") from self._th_err

    def _th_run_train(self) -> None:
        if self._th_train is not None:
            return
        with self._lock:
            if self._th_train is not None:
                return

            def is_alive() -> bool:
                return self._th_train is th and not self._th_term

            def run() -> None:
                try:
                    lplan = self._epoch_learning_plan(
                        self._cur_epoch, self._train_learning_plan)
                    self._th_compute_batch(
                        is_alive, self._train, lplan, self._train_buff)
                except BaseException as e:  # pylint: disable=broad-except
                    self._th_err = e
                finally:
                    with self._lock:
                        if self._th_train is th:
                            self._th_train = None
                    with self._cond:
                        self._cond.notify_all()

            th = threading.Thread(target=run, daemon=True)
            self._th_train = th
            th.start()

    def _th_run_train_val(self) -> None:
        if self._th_train_val is not None:
            return
        with self._lock:
            if self._th_train_val is not None:
                return

            def is_alive() -> bool:
                return self._th_train_val is th and not self._th_term

            def run() -> None:
                try:
                    lplan = self._epoch_learning_plan(
                        0, self._train_val_learning_plan)
                    self._th_compute_batch(
                        is_alive,
                        self._train_validation,
                        lplan,
                        self._train_validation_buff)
                except BaseException as e:  # pylint: disable=broad-except
                    self._th_err = e
                finally:
                    with self._lock:
                        if self._th_train_val is th:
                            self._th_train_val = None
                    with self._cond:
                        self._cond.notify_all()

            th = threading.Thread(target=run, daemon=True)
            self._th_train_val = th
            th.start()

    def _th_run_test(self) -> None:
        if self._th_test is not None:
            return
        with self._lock:
            if self._th_test is not None:
                return

            def is_alive() -> bool:
                return self._th_test is th and not self._th_term

            def run() -> None:
                try:
                    lplan = self._epoch_learning_plan(
                        0, self._test_learning_plan)
                    self._th_compute_batch(
                        is_alive,
                        self._test,
                        lplan,
                        self._test_buff)
                except BaseException as e:  # pylint: disable=broad-except
                    self._th_err = e
                finally:
                    with self._lock:
                        if self._th_test is th:
                            self._th_test = None
                    with self._cond:
                        self._cond.notify_all()

            th = threading.Thread(target=run, daemon=True)
            self._th_test = th
            th.start()

    def _th_run_test_val(self) -> None:
        if self._th_test_val is not None:
            return
        with self._lock:
            if self._th_test_val is not None:
                return

            def is_alive() -> bool:
                return self._th_test_val is th and not self._th_term

            def run() -> None:
                try:
                    lplan = self._epoch_learning_plan(
                        0, self._test_val_learning_plan)
                    self._th_compute_batch(
                        is_alive,
                        self._test_validation,
                        lplan,
                        self._test_validation_buff)
                except BaseException as e:  # pylint: disable=broad-except
                    self._th_err = e
                finally:
                    with self._lock:
                        if self._th_test_val is th:
                            self._th_test_val = None
                    with self._cond:
                        self._cond.notify_all()

            th = threading.Thread(target=run, daemon=True)
            self._th_test_val = th
            th.start()

    def next_train_batch(self) -> list[BatchRow] | None:
        train_size = self.get_epoch_train_size()
        if self._cur_train_ix >= train_size:
            return None
        cache = self._cur_train_cache
        while self._cur_train_ix >= len(cache):
            self._get_batch_for(self._th_run_train, self._train_buff, cache)
        end_ix = min(
            self._cur_train_ix + self._batch_size, train_size)
        res = cache[self._cur_train_ix:end_ix]
        self._cur_train_ix += len(res)
        return res

    def next_train_validation_batch(self) -> list[BatchRow] | None:
        train_val_size = self._train_val_size
        if self._cur_train_validation_ix >= train_val_size:
            return None
        cache = self._cur_train_validation_cache
        while self._cur_train_validation_ix >= len(cache):
            self._get_batch_for(
                self._th_run_train_val, self._train_validation_buff, cache)
        end_ix = min(
            self._cur_train_validation_ix + self._batch_size, train_val_size)
        res = cache[self._cur_train_validation_ix:end_ix]
        self._cur_train_validation_ix += len(res)
        return res

    def next_test_batch(self) -> list[BatchRow] | None:
        test_size = self._test_size
        if self._cur_test_ix >= test_size:
            return None
        cache = self._cur_test_cache
        while self._cur_test_ix >= len(cache):
            self._get_batch_for(self._th_run_test, self._test_buff, cache)
        end_ix = min(
            self._cur_test_ix + self._batch_size, test_size)
        res = cache[self._cur_test_ix:end_ix]
        self._cur_test_ix += len(res)
        return res

    def next_test_validation_batch(self) -> list[BatchRow] | None:
        test_val_size = self._test_val_size
        if self._cur_test_validation_ix >= test_val_size:
            return None
        cache = self._cur_test_validation_cache
        while self._cur_test_validation_ix >= len(cache):
            self._get_batch_for(
                self._th_run_test_val, self._test_validation_buff, cache)
        end_ix = min(
            self._cur_test_validation_ix + self._batch_size, test_val_size)
        res = cache[self._cur_test_validation_ix:end_ix]
        self._cur_test_validation_ix += len(res)
        return res

    def train_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_train_ix >= self.get_epoch_train_size():
            raise ValueError("train batches already exhausted!")
        while True:
            res = self.next_train_batch()
            if res is None:
                return
            yield res

    def train_validation_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_train_validation_ix >= self._train_val_size:
            raise ValueError("train validation batches already exhausted!")
        while True:
            res = self.next_train_validation_batch()
            if res is None:
                return
            yield res

    def test_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_test_ix >= self._test_size:
            raise ValueError("test batches already exhausted!")
        while True:
            res = self.next_test_batch()
            if res is None:
                return
            yield res

    def test_validation_batches(self) -> Iterable[list[BatchRow]]:
        if self._cur_test_validation_ix >= self._test_val_size:
            raise ValueError("test validation batches already exhausted!")
        while True:
            res = self.next_test_validation_batch()
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

    def test_validation_dfs(self) -> Iterable[pd.DataFrame]:
        yield from (
            pd.DataFrame(val, columns=COLUMNS)
            for val in self.test_validation_batches()
        )


def create_train_test(
        *,
        train_ns: Namespace,
        train_validation_ns: Namespace,
        test_ns: Namespace,
        test_validation_ns: Namespace,
        train_learning_plan: Sequence[EpochLearningPlan | LearningPlan],
        train_val_learning_plan: list[LearningPlan],
        test_learning_plan: list[LearningPlan],
        test_val_learning_plan: list[LearningPlan],
        batch_size: int,
        epoch_batches: int,
        train_val_size: int,
        test_size: int,
        test_val_size: int,
        train_seed: int = 42,
        train_validation_seed: int = 37,
        test_seed: int = 69,
        test_validation_seed: int = 23,
        compute_batch_size: int | None = None,
        scorer: Scorer | None = None,
        now: pd.Timestamp | None = None) -> TrainTestGenerator:
    return TrainTestGenerator(
        train=DataGenerator(train_ns, train_seed),
        train_validation=DataGenerator(
            train_validation_ns, train_validation_seed),
        test=DataGenerator(test_ns, test_seed),
        test_validation=DataGenerator(
            test_validation_ns, test_validation_seed),
        train_learning_plan=train_learning_plan,
        train_val_learning_plan=train_val_learning_plan,
        test_learning_plan=test_learning_plan,
        test_val_learning_plan=test_val_learning_plan,
        batch_size=batch_size,
        epoch_batches=epoch_batches,
        train_val_size=train_val_size,
        test_size=test_size,
        test_val_size=test_val_size,
        compute_batch_size=compute_batch_size,
        scorer=scorer,
        now=now)
