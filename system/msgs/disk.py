from typing import TYPE_CHECKING, Iterable
import os
import weakref

from misc.env import envload_path
from misc.io import open_append, open_read
from system.msgs.message import MHash, Message
from system.msgs.store import MessageStore


if TYPE_CHECKING:
    WVD = weakref.WeakValueDictionary[MHash, Message]
else:
    WVD = weakref.WeakValueDictionary


class DiskStore(MessageStore):
    def __init__(self) -> None:
        self._path = envload_path("MSG_PATH")
        self._topics = envload_path("MSG_TOPICS")
        self._cache: WVD = weakref.WeakValueDictionary()

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n")

    @staticmethod
    def _unescape(text: str) -> str:
        return text.replace("\\n", "\n").replace("\\\\", "\\")

    def _compute_path(self, message_hash: MHash) -> str:

        def split_hash(hash: str) -> Iterable[str]:
            yield hash[:2]
            yield hash[2:4]
            yield hash[4:6]
            yield hash[6:56]
            yield hash[56:]

        all_segs = list(split_hash(message_hash.get_raw_hash()))
        segs = all_segs[:-1]
        rest = all_segs[-1]
        return os.path.join(self._path, *segs, f"{rest}.msg")

    def write_message(self, message: Message) -> MHash:
        assert message.is_valid_message()
        mhash = message.get_hash()
        with open_append(self._compute_path(mhash), text=True) as fout:
            fout.write(f"{self._escape(message.get_text())}\n")
        return message.get_hash()

    def read_message(self, message_hash: MHash) -> Message:
        res = self._cache.get(message_hash, None)
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
                    self._cache[mhash] = msg
                    if mhash == message_hash:
                        res = msg
        except FileNotFoundError:
            pass
        if res is None:
            raise KeyError(f"no message for the hash: {message_hash}")
        return res

    def add_topic(self, topic: Message) -> MHash:
        assert topic.is_topic()
        with open_append(self._topics, text=True) as fout:
            fout.write(f"{self._escape(topic.get_text())}\n")
        return topic.get_hash()

    def get_topics(self) -> Iterable[Message]:
        with open_read(self._topics, text=True) as fin:
            for line in fin:
                line = line.rstrip()
                if not line:
                    continue
                text = self._unescape(line)
                msg = Message(msg=text)
                assert msg.is_topic()
                yield msg
