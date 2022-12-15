from misc.redis import set_redis_slow_mode
from model.datagenerator import DataGenerator
from system.namespace.store import get_namespace


if __name__ == "__main__":
    set_redis_slow_mode("never")
    ns = get_namespace("test")
    data_gen = DataGenerator(ns, 42)
    for link in data_gen.get_random_links(100):
        print(
            f"{data_gen.short_info(link.get_parent())} -- "
            f"{data_gen.short_info(link.get_child())} -- "
            f"{data_gen.vote_score(link)}")
