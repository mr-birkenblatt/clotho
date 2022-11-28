from typing import Iterable

import numpy as np

from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


class RamMessageStore(MessageStore):
    def __init__(self) -> None:
        self._msgs: dict[MHash, Message] = {}
        self._topics: list[Message] = []

    def write_message(self, message: Message) -> MHash:
        mhash = message.get_hash()
        self._msgs[mhash] = message
        return mhash

    def read_message(self, message_hash: MHash) -> Message:
        return self._msgs[message_hash]

    def add_topic(self, topic: Message) -> MHash:
        self._topics.append(topic)
        return topic.get_hash()

    def get_topics(self, offset: int, limit: int) -> Iterable[Message]:
        yield from self._topics[offset:offset + limit]

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        keys = list(self._msgs.keys())
        yield from (keys[ix] for ix in rng.choice(len(keys), size=count))
