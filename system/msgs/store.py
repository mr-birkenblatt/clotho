from typing import Iterable, Literal, TypedDict

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
            limit: int | None) -> Iterable[Message]:
        raise NotImplementedError()

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        raise NotImplementedError()

    def get_random_messages(
            self,
            ref: MHash,
            is_parent: bool,
            offset: int,
            limit: int) -> Iterable[MHash]:
        start = offset - (offset % RNG_ALIGN)
        end = offset + limit
        base_seed = hash(ref)
        if is_parent:
            base_seed = base_seed * 5 + 1
        res: list[MHash] = []
        cur_ix = start
        while cur_ix < end:
            rng = np.random.default_rng(abs(base_seed + SEED_MUL * cur_ix))
            res.extend(self.do_get_random_messages(rng, RNG_ALIGN))
            cur_ix += RNG_ALIGN
        rel_start = offset - start
        return res[rel_start:rel_start + limit]


MSG_STORE: dict[Namespace, MessageStore] = {}


def get_message_store(namespace: Namespace) -> MessageStore:
    res = MSG_STORE.get(namespace)
    if res is None:
        res = create_message_store(namespace.get_message_module())
        MSG_STORE[namespace] = res
    return res


MessageStoreName = Literal["disk", "ram"]


MsgsModule = TypedDict('MsgsModule', {
    "name": MessageStoreName,
    "root": str,
})


def create_message_store(mobj: MsgsModule) -> MessageStore:
    name = mobj["name"]
    if name == "disk":
        from system.msgs.disk import DiskStore
        return DiskStore(mobj["root"])
    if name == "ram":
        from system.msgs.ram import RamMessageStore
        return RamMessageStore()
    raise ValueError(f"unknown message store: {name}")
