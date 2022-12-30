from typing import Callable, Iterable, Literal, TypedDict

import numpy as np

from system.msgs.message import Message, MHash
from system.namespace.namespace import Namespace


RNG_ALIGN = 10
SEED_MUL = 17


class MessageStore:
    def write_message(self, message: Message) -> MHash:
        raise NotImplementedError()

    def read_message(self, message_hash: MHash) -> Message:
        raise NotImplementedError()

    def add_topic(self, topic: Message) -> MHash:
        raise NotImplementedError()

    def get_topics(
            self,
            offset: int,
            limit: int | None) -> list[Message]:
        raise NotImplementedError()

    def get_topics_count(self) -> int:
        return len(self.get_topics(0, None))

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        raise NotImplementedError()

    def get_random_messages(
            self,
            ref: MHash,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        base_seed = hash(ref)
        if is_parent:
            base_seed = base_seed * 5 + 1
        yield from self.generate_random_messages(
            lambda cur_ix: np.random.default_rng(
                abs(base_seed + SEED_MUL * cur_ix)),
            offset,
            limit)

    def generate_random_messages(
            self,
            rng_fn: Callable[[int], np.random.Generator],
            offset: int,
            limit: int) -> list[MHash]:
        start = offset - (offset % RNG_ALIGN)
        end = offset + limit
        res: list[MHash] = []
        cur_ix = start
        while cur_ix < end:
            rng = rng_fn(cur_ix)
            res.extend(self.do_get_random_messages(rng, RNG_ALIGN))
            cur_ix += RNG_ALIGN
        rel_start = offset - start
        return res[rel_start:rel_start + limit]

    def enumerate_messages(self, progress_bar: bool) -> Iterable[MHash]:
        raise NotImplementedError()


MSG_STORE: dict[Namespace, MessageStore] = {}


def get_message_store(namespace: Namespace) -> MessageStore:
    res = MSG_STORE.get(namespace)
    if res is None:
        res = create_message_store(namespace.get_message_module())
        MSG_STORE[namespace] = res
    return res


DiskMessageModule = TypedDict('DiskMessageModule', {
    "name": Literal["disk"],
    "root": str,
})
RamMessageModule = TypedDict('RamMessageModule', {
    "name": Literal["ram"],
})
MsgsModule = DiskMessageModule | RamMessageModule


def create_message_store(mobj: MsgsModule) -> MessageStore:
    if mobj["name"] == "disk":
        from system.msgs.disk import DiskStore
        return DiskStore(mobj["root"])
    if mobj["name"] == "ram":
        from system.msgs.ram import RamMessageStore
        return RamMessageStore()
    raise ValueError(f"unknown message store: {mobj}")
