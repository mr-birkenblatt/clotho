import pandas as pd

from misc.redis import set_redis_slow_mode
from model.datagenerator import create_train_test, DataGenerator
from system.namespace.store import get_namespace


MESSAGE_GENERATION = False


def run() -> None:
    namespace = get_namespace("test")
    if MESSAGE_GENERATION:
        data_gen = DataGenerator(namespace, 42)
        for link in data_gen.get_valid_random_links(
                100, conversation_based=True):
            print(
                f"{data_gen.short_info(link.get_parent())} -- "
                f"{data_gen.short_info(link.get_child())} -- "
                f"{data_gen.vote_score(link)}")
        print("====================")
        for link in data_gen.get_valid_random_links(
                5, conversation_based=True):
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
        ttgen = create_train_test(
            train_ns=ns_train,
            train_validation_ns=ns_train,
            test_ns=namespace,
            test_validation_ns=namespace,
            batch_size=4,
            epoch_batches=100,
            train_val_size=500,
            test_size=500,
            test_val_size=500,
            compute_batch_size=10,
            conversation_based=False)
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
