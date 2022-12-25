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
from system.namespace.store import get_namespace


MESSAGE_GENERATION = False


def run() -> None:
    namespace = get_namespace("test")
    if MESSAGE_GENERATION:
        data_gen = DataGenerator(namespace, 42)
        for link in data_gen.get_valid_random_links(
                100, scorer=get_scorer("best"), now=now_ts()):
            print(
                f"{data_gen.short_info(link.get_parent())} -- "
                f"{data_gen.short_info(link.get_child())} -- "
                f"{data_gen.vote_score(link)}")
        print("====================")
        for link in data_gen.get_valid_random_links(
                5, scorer=get_scorer("best"), now=now_ts()):
            print(f"{data_gen.long_info(link.get_parent())}")
            print("--------------------")
            print(f"{data_gen.long_info(link.get_child())}")
            print("--------------------")
            print(f"{data_gen.vote_score(link)}")
            print("====================")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        ns_train = get_namespace("train")
        train_plan: list[EpochLearningPlan] = [
            {
                "left": {"mode": "valid", "flip_pc": 0.5},
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "first_epoch": 10,
                "last_epoch": None,
                "weight": 100,
            },
            {
                "left": {"mode": "random", "flip_pc": 0.0},
                "right": {"mode": "path", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": 5,
                "weight": 60,
            },
            {
                "left": None,
                "right": {"mode": "path", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": 5,
                "weight": 40,
            },
            {
                "left": {"mode": "random", "flip_pc": 0.0},
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": None,
                "weight": 60,
            },
            {
                "left": None,
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "first_epoch": None,
                "last_epoch": None,
                "weight": 40,
            }
        ]
        eval_plan: list[LearningPlan] = [
            {
                "left": {"mode": "random", "flip_pc": 0.0},
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "weight": 60,
            },
            {
                "left": None,
                "right": {"mode": "valid", "flip_pc": 0.0},
                "min_text_length": 20,
                "flip_lr": 0.5,
                "weight": 40,
            }
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
        for _ in range(4):
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
    run()
