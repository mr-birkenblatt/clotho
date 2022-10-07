from typing import Dict, Iterable, List

from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


class RamMessageStore(MessageStore):
    def __init__(self) -> None:
        self._msgs: Dict[MHash, Message] = {}
        self._topics: List[Message] = []

    def write_message(self, message: Message) -> MHash:
        mhash = message.get_hash()
        self._msgs[mhash] = message
        return mhash

    def read_message(self, message_hash: MHash) -> Message:
        return self._msgs[message_hash]

    def add_topic(self, topic: Message) -> MHash:
        self._topics.append(topic)
        return topic.get_hash()

    def get_topics(self) -> Iterable[Message]:
        yield from self._topics