import os
from typing import Iterable

from misc.env import envload_path
from misc.io import ensure_folder, open_append, open_read
from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


class DiskStore(MessageStore):
    def __init__(self) -> None:
        self._path = envload_path("MSG_PATH", default="userdata/msg")
        self._topics = envload_path(
            "MSG_TOPICS", default="userdata/topics.list")
        self._cache: LRU[MHash, Message] = LRU(10000)

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n")

    @staticmethod
    def _unescape(text: str) -> str:
        return text.replace("\\n", "\n").replace("\\\\", "\\")

    def _compute_path(self, message_hash: MHash) -> str:

        def split_hash(hash_str: str) -> Iterable[str]:
            yield hash_str[:2]
            yield hash_str[2:4]
            yield hash_str[4:6]
            yield hash_str[6:56]
            # NOTE: we ignore the last segment
            # yield hash_str[56:]

        all_segs = list(split_hash(message_hash.to_parseable()))
        segs = all_segs[:-1]
        rest = all_segs[-1]
        return os.path.join(self._path, *segs, f"{rest}.msg")

    def write_message(self, message: Message) -> MHash:
        assert message.is_valid_message()
        mhash = message.get_hash()
        ensure_folder(os.path.dirname(self._compute_path(mhash)))
        with open_append(self._compute_path(mhash), text=True) as fout:
            fout.write(f"{self._escape(message.get_text())}\n")
        return message.get_hash()

    def read_message(self, message_hash: MHash) -> Message:
        res = self._cache.get(message_hash)
        if res is not None:
            return res
        try:
            with open_read(self._compute_path(message_hash), text=True) as fin:
                for line in fin:
                    line = line.rstrip()
                    if not line:
                        continue
                    text = self._unescape(line)
                    msg = Message(msg=text)
                    mhash = msg.get_hash()
                    self._cache.set(mhash, msg)
                    if mhash == message_hash:
                        res = msg
        except FileNotFoundError:
            pass
        if res is None:
            raise KeyError(f"no message for the hash: {message_hash}")
        return res

    def add_topic(self, topic: Message) -> MHash:
        if not topic.is_topic():
            raise ValueError(f"{topic}(\"{topic.get_text()}\") is not a topic")
        with open_append(self._topics, text=True) as fout:
            fout.write(f"{self._escape(topic.get_text())}\n")
        return topic.get_hash()

    def get_topics(self) -> Iterable[Message]:
        try:
            with open_read(self._topics, text=True) as fin:
                for line in fin:
                    line = line.rstrip()
                    if not line:
                        continue
                    text = self._unescape(line)
                    msg = Message(msg=text)
                    assert msg.is_topic()
                    yield msg
        except FileNotFoundError:
            pass
