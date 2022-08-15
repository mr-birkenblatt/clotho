from typing import Iterable, Optional

from misc.env import envload_str
from system.msgs.message import Message, MHash


class MessageStore:
    def write_message(self, message: Message) -> MHash:
        raise NotImplementedError()

    def read_message(self, message_hash: MHash) -> Message:
        raise NotImplementedError()

    def add_topic(self, topic: Message) -> MHash:
        raise NotImplementedError()

    def get_topics(self) -> Iterable[Message]:
        raise NotImplementedError()


DEFAULT_MSG_STORE: Optional[MessageStore] = None


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
    raise ValueError(f"unknown message store: {name}")
