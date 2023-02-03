import sys
import time

import numpy as np
import pandas as pd

from misc.redis import set_redis_slow_mode
from misc.util import now_ts
from model.datagenerator import (
    create_train_test,
    DataGenerator,
    EpochLearningPlan,
    LearningPlan,
)
from system.links.scorer import get_scorer

# from system.logger.frontend import register_logger_backend
from system.msgs.store import get_message_store
from system.namespace.store import get_namespace


RANDOM_TEST = False
MESSAGE_GENERATION = False
GENERATE_ALL = True


def run(ns_name: str, ns_name_other: str | None) -> None:
    namespace = get_namespace(ns_name)
    if GENERATE_ALL:
        # register_logger_backend("stdcount")
        now = now_ts()
        data_gen = DataGenerator(namespace, 42)
        cur_time = time.monotonic()
        valid_score_links = 0
        valid_links = 0
        for link in data_gen.get_all_valid_links(now, progress_bar=True):
            valid_links += 1
            if link.get_votes("up").get_total_votes() >= 2.0:
                valid_score_links += 1
        print(f"valid links: {valid_links}")
        print(f"valid score links: {valid_score_links}")
        path_links = 0
        for _ in data_gen.get_all_path_links(now):
            path_links += 1
        print(f"valid links: {path_links}")
        print(f"time: {time.monotonic() - cur_time:.4f}s")
    elif RANDOM_TEST:
        msgs = get_message_store(namespace)
        cur_time = time.monotonic()
        for mhash in msgs.generate_random_messages(
                lambda seed: np.random.default_rng(42 * seed + 23), 0, 100):
            print(f"{mhash}: {msgs.read_message(mhash).get_text()[:40]}")
        print(f"time: {time.monotonic() - cur_time:.4f}s")
    elif MESSAGE_GENERATION:
        data_gen = DataGenerator(namespace, 42)
        for link in data_gen.get_valid_random_links(
                100, scorer=get_scorer("best"), now=now_ts(), verbose=False):
            print(
                f"{data_gen.short_info(link.get_parent())} -- "
                f"{data_gen.short_info(link.get_child())} -- "
                f"{data_gen.vote_score(link)}")
        print("====================")
        for link in data_gen.get_valid_random_links(
                5, scorer=get_scorer("best"), now=now_ts(), verbose=False):
            print(f"{data_gen.long_info(link.get_parent())}")
            print("--------------------")
            print(f"{data_gen.long_info(link.get_child())}")
            print("--------------------")
            print(f"{data_gen.vote_score(link)}")
            print("====================")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        ns_train = (
            namespace
            if ns_name_other is None
            else get_namespace(ns_name_other))
        train_plan: list[EpochLearningPlan] = [
            {
                "left": {"mode": "valid", "flip_pc": 0.5},
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": None,
                "skip_weak": True,
                "skip_topics": True,
                "flip_lr": 0.5,
                "first_epoch": 10,
                "last_epoch": None,
                "weight": 100,
            },
            {
                "left": {"mode": "random", "flip_pc": 0.0},
                "right": {"mode": "path", "flip_pc": 0.0},
                "min_text_length": None,
                "skip_weak": True,
                "skip_topics": True,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": 5,
                "weight": 60,
            },
            {
                "left": None,
                "right": {"mode": "path", "flip_pc": 0.0},
                "min_text_length": 20,
                "skip_weak": False,
                "skip_topics": True,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": 5,
                "weight": 40,
            },
            {
                "left": {"mode": "random", "flip_pc": 0.0},
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "skip_weak": False,
                "skip_topics": True,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": None,
                "weight": 60,
            },
            {
                "left": None,
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "skip_weak": False,
                "skip_topics": True,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": None,
                "weight": 40,
            },
        ]
        eval_plan: list[LearningPlan] = [
            {
                "left": {"mode": "random", "flip_pc": 0.0},
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "skip_weak": False,
                "skip_topics": True,
                "flip_lr": 0.5,
                "weight": 60,
            },
            {
                "left": None,
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "skip_weak": False,
                "skip_topics": True,
                "flip_lr": 0.5,
                "weight": 40,
            },
        ]
        ttgen = create_train_test(
            train_ns=ns_train,
            train_validation_ns=ns_train,
            test_ns=namespace,
            test_validation_ns=namespace,
            train_learning_plan=train_plan,
            train_val_learning_plan=eval_plan,
            test_learning_plan=eval_plan,
            test_val_learning_plan=eval_plan,
            batch_size=4,
            epoch_batches=100,
            train_val_size=500,
            test_size=500,
            test_val_size=500,
            compute_batch_size=10)
        ttgen.set_epoch(3)
        bar = "=" * 42
        for cur_iter in range(5):
            if cur_iter == 4:
                ttgen.set_epoch(3)
            epoch = ttgen.get_epoch()
            print(bar)
            print(f"train {epoch}")
            print(bar)
            size = 0
            last_df = None
            for train_df in ttgen.train_dfs():
                size += train_df.shape[0]
                print(train_df if last_df is None else size)
                last_df = train_df
            if last_df is not None:
                print(last_df)

            print(bar)
            print(f"train validation {epoch}")
            print(bar)
            size = 0
            last_df = None
            for train_validation_df in ttgen.train_validation_dfs():
                size += train_validation_df.shape[0]
                print(train_validation_df if last_df is None else size)
                last_df = train_validation_df
            if last_df is not None:
                print(last_df)

            print(bar)
            print(f"test {epoch}")
            print(bar)
            size = 0
            last_df = None
            for test_df in ttgen.test_dfs():
                size += test_df.shape[0]
                print(test_df if last_df is None else size)
                last_df = test_df
            if last_df is not None:
                print(last_df)
            ttgen.advance_epoch()

        print(bar)
        print("test validation")
        print(bar)
        size = 0
        last_df = None
        for test_validation_df in ttgen.test_validation_dfs():
            size += test_validation_df.shape[0]
            print(test_validation_df if last_df is None else size)
            last_df = test_validation_df
        if last_df is not None:
            print(last_df)


if __name__ == "__main__":
    set_redis_slow_mode("never")
    run(sys.argv[1], None if len(sys.argv) < 3 else sys.argv[2])
