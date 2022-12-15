from model.datagenerator import DataGenerator
from system.namespace.store import get_namespace


if __name__ == "__main__":
    ns = get_namespace("test")
    data_gen = DataGenerator(ns, 42)
    for link in data_gen.get_random_links(100):
        print(
            f"{data_gen.long_info(link.get_parent())} -- "
            f"{data_gen.long_info(link.get_child())} -- "
            f"{data_gen.vote_score(link)}")
