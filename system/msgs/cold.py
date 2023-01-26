import os
from typing import Iterable

import numpy as np

from misc.cold_writer import ColdAccess
from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


class ColdMessageStore(MessageStore):
    def __init__(self, msgs_root: str, *, keep_alive: float) -> None:
        super().__init__()
        self._msgs = ColdAccess(
            os.path.join(msgs_root, "msgs.zip"), keep_alive=keep_alive)
        self._topics = ColdAccess(
            os.path.join(msgs_root, "topics.zip"), keep_alive=keep_alive)
        self._buff: LRU[MHash, Message] = LRU(100)

    def write_message(self, message: Message) -> MHash:
        self._msgs.write_line(message.get_text())
        res = message.get_hash()
        self._buff.set(res, message)
        return res

    def read_message(self, message_hash: MHash) -> Message:
        res = self._buff.get(message_hash)
        if res is not None:
            return res
        raise RuntimeError(
            "reading specific messages is not supported in cold storage")

    def add_topic(self, topic: Message) -> MHash:
        if not topic.is_topic():
            raise ValueError(f"{topic}(\"{topic.get_text()}\") is not a topic")
        self._topics.write_line(topic.get_text())
        return topic.get_hash()

    def _get_topics(self) -> Iterable[Message]:
        for line in self._topics.enumerate_lines():
            yield Message(msg=line)

    def get_topics_count(self) -> int:
        count = 0
        for _ in self._topics.enumerate_lines():
            count += 1
        return count

    def get_topics(
            self,
            offset: int,
            limit: int | None) -> list[Message]:
        if limit is None:
            end_off = None
        else:
            end_off = offset + limit
        res = []
        for ix, msg in enumerate(self._get_topics()):
            if ix < offset:
                continue
            if end_off is not None and ix >= end_off:
                break
            res.append(msg)
        return res

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        raise RuntimeError(
            "retrieving random messages is not supported in cold storage")

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:
        for line in self._msgs.enumerate_lines():
            msg = Message(msg=line)
            res = msg.get_hash()
            self._buff.set(res, msg)
            yield res

    def get_message_count(self) -> int:
        count = 0
        for _ in self._msgs.enumerate_lines():
            count += 1
        return count
