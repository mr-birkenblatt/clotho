# from typing import Callable, Literal
import numpy as np

from system.links.store import get_link_store
from system.msgs.message import MHash
from system.msgs.store import get_message_store
from system.namespace.namespace import Namespace


SEED_MUL = 17
PATH_PROB = 0.5


class DataGenerator:
    def __init__(self, namespace: Namespace, seed: int) -> None:
        self._msgs = get_message_store(namespace)
        self._links = get_link_store(namespace)
        self._seed = seed
        self._rng = np.random.default_rng(abs(seed))
        self._rix = 0

    def reset(self) -> None:
        seed = self._seed
        self._rng = np.random.default_rng(abs(seed))
        self._rix = 0

    def get_random_message(self, count: int) -> list[MHash]:
        seed = self._seed
        rix = self._rix
        self._rix += count

        def get_rng(cur_ix: int) -> np.random.Generator:
            return np.random.default_rng(abs(seed + SEED_MUL * cur_ix))

        return self._msgs.generate_random_messages(
            get_rng, rix, count)

    def get_random_paths(self, count: int) -> list[list[int]]:
        rng = self._rng

        def get_path() -> list[int]:
            cur = []
            cur_ix = 0
            while True:
                if rng.random() < PATH_PROB:
                    cur_ix += 1
                elif rng.random() < PATH_PROB:
                    cur.append(cur_ix)
                    cur_ix = 0
                else:
                    return cur

        return [get_path() for _ in range(count)]

    # def get_link_from_paths(self, paths: list[list[int]]) -> list[int]:
    #     msgs = self._msgs
    #     links = self._links

    #     def get_for(ix: int, at: Callable[[int], MHash | None]) -> MHash:
    #         res = None
    #         while res is None:
    #             res = at(ix)
    #             ix -= 1

    #     def get_link(path: list[int]) -> int:
    #         msgs.get_topics(path[0], 1)
    #         while ix < path:

    #             while path[ix] == "right":
    #                 ix += 1

    #     return [get_link(path) for path in paths]

    # def get_random_links(self, )
