from typing import Callable, Iterable, Literal, TypedDict

import numpy as np

from system.msgs.message import Message, MHash
from system.namespace.module import ModuleBase
from system.namespace.namespace import ModuleName, Namespace


RNG_ALIGN = 10
SEED_MUL = 17


class MessageStore(ModuleBase):
    @staticmethod
    def module_name() -> ModuleName:
        return "msgs"

    def from_namespace(
            self, other_namespace: Namespace, *, progress_bar: bool) -> None:
        omsgs = get_message_store(other_namespace)
        for topic in omsgs.get_topics(offset=0, limit=None):
            self.add_topic(topic)
        for msg in omsgs.enumerate_messages(progress_bar=progress_bar):
            self.write_message(omsgs.read_message(msg))

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

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:
        raise NotImplementedError()

    def get_message_count(self) -> int:
        raise NotImplementedError()


MSG_STORE: dict[Namespace, MessageStore] = {}


def get_message_store(namespace: Namespace) -> MessageStore:
    res = MSG_STORE.get(namespace)
    if res is None:
        res = create_message_store(namespace)
        MSG_STORE[namespace] = res
    return res


ColdMessageModule = TypedDict('ColdMessageModule', {
    "name": Literal["cold"],
    "keep_alive": float,
})
DiskMessageModule = TypedDict('DiskMessageModule', {
    "name": Literal["disk"],
    "cache_size": int,
})
DBMessageModule = TypedDict('DBMessageModule', {
    "name": Literal["db"],
    "conn": str,
    "cache_size": int,
})
RamMessageModule = TypedDict('RamMessageModule', {
    "name": Literal["ram"],
})
MsgsModule = (
    ColdMessageModule |
    DiskMessageModule |
    DBMessageModule |
    RamMessageModule
)


def create_message_store(namespace: Namespace) -> MessageStore:
    mobj = namespace.get_message_module()
    if mobj["name"] == "disk":
        from system.msgs.disk import DiskStore
        return DiskStore(namespace.get_root(), mobj["cache_size"])
    if mobj["name"] == "db":
        from system.msgs.db import DBStore
        return DBStore(
            namespace,
            namespace.get_db_connector(mobj["conn"]),
            mobj["cache_size"])
    if mobj["name"] == "ram":
        from system.msgs.ram import RamMessageStore
        return RamMessageStore()
    if mobj["name"] == "cold":
        from system.msgs.cold import ColdMessageStore
        return ColdMessageStore(
            namespace.get_module_root("msgs"), keep_alive=mobj["keep_alive"])
    raise ValueError(f"unknown message store: {mobj}")
