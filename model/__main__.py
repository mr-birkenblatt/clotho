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
            batch_size=4,
            epoch_batches=100,
            train_val_size=500,
            test_size=500,
            compute_batch_size=10,
            conversation_based=False)
        for _ in range(4):
            epoch = ttgen.get_epoch()
            print(f"train {epoch}")
            first = True
            size = 0
            last_df = None
            for train_df in ttgen.train_dfs():
                size += train_df.shape[0]
                print(train_df if first else size)
                last_df = train_df
                first = False
            if last_df is not None:
                print(last_df)
            print(f"train validation {epoch}")
            first = True
            size = 0
            last_df = None
            for train_validation_df in ttgen.train_validation_dfs():
                size += train_validation_df.shape[0]
                print(train_validation_df if first else size)
                last_df = train_validation_df
                first = False
            if last_df is not None:
                print(last_df)
            print(f"test {epoch}")
            first = True
            size = 0
            last_df = None
            for test_df in ttgen.test_dfs():
                size += test_df.shape[0]
                print(test_df if first else size)
                last_df = test_df
                first = False
            if last_df is not None:
                print(last_df)
            ttgen.advance_epoch()


if __name__ == "__main__":
    set_redis_slow_mode("never")
    run()
