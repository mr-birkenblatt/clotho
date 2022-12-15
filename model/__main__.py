import pandas as pd

from misc.redis import set_redis_slow_mode
from model.datagenerator import create_train_test, DataGenerator
from system.namespace.store import get_namespace


MESSAGE_GENERATION = False


if __name__ == "__main__":
    set_redis_slow_mode("never")
    ns = get_namespace("test")
    if MESSAGE_GENERATION:
        data_gen = DataGenerator(ns, 42)
        for link in data_gen.get_valid_random_links(100):
            print(
                f"{data_gen.short_info(link.get_parent())} -- "
                f"{data_gen.short_info(link.get_child())} -- "
                f"{data_gen.vote_score(link)}")
        print("====================")
        for link in data_gen.get_valid_random_links(5):
            print(f"{data_gen.long_info(link.get_parent())}")
            print("--------------------")
            print(f"{data_gen.long_info(link.get_child())}")
            print("--------------------")
            print(f"{data_gen.vote_score(link)}")
            print("====================")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        ttgen = create_train_test(
            train_ns=ns,
            train_validation_ns=ns,
            test_ns=ns,
            batch_size=3,
            epoch_batches=2)
        for _ in range(4):
            epoch = ttgen.get_epoch()
            print(f"train {epoch}")
            for train_df in ttgen.train_dfs():
                print(train_df)
            print(f"train validation {epoch}")
            for train_validation_df in ttgen.train_validation_dfs():
                print(train_validation_df)
            print(f"test {epoch}")
            for test_df in ttgen.test_dfs():
                print(test_df)
            ttgen.advance_epoch()
