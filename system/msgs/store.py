from typing import Iterable

import numpy as np

from misc.env import envload_str
from system.msgs.message import Message, MHash


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
            ref: MHash | None,
            offset: int,
            limit: int) -> Iterable[MHash]:
        start = offset - (offset % RNG_ALIGN)
        end = offset + limit
        base_seed = 1 if ref is None else hash(ref)
        res: list[MHash] = []
        cur_ix = start
        while cur_ix < end:
            rng = np.random.default_rng(abs(base_seed + SEED_MUL * cur_ix))
            res.extend(self.do_get_random_messages(rng, RNG_ALIGN))
            cur_ix += RNG_ALIGN
        rel_start = offset - start
        return res[rel_start:rel_start + limit]


DEFAULT_MSG_STORE: MessageStore | None = None


def get_default_message_store() -> MessageStore:
    global DEFAULT_MSG_STORE

    if DEFAULT_MSG_STORE is None:
        DEFAULT_MSG_STORE = get_message_store(
            envload_str("MSG_STORE", default="disk"))
    return DEFAULT_MSG_STORE


def get_message_store(name: str) -> MessageStore:
    if name == "disk":
        from system.msgs.disk import DiskStore
        return DiskStore()
    if name == "ram":
        from system.msgs.ram import RamMessageStore
        return RamMessageStore()
    raise ValueError(f"unknown message store: {name}")
