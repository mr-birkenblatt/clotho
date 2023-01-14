import os
import time
from typing import Callable, Iterable

import numpy as np

from misc.io import ensure_folder, get_folder, open_append, open_read
from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


MSG_EXT = ".msg"
RELOAD_TOPICS_FREQ = 60 * 60  # 1h


class DiskStore(MessageStore):
    def __init__(self, msgs_root: str, cache_size: int) -> None:
        self._path = os.path.join(msgs_root, "msg")
        self._topics = os.path.join(msgs_root, "topics.list")
        self._cache: LRU[MHash, Message] = LRU(cache_size)
        self._topic_cache: list[Message] | None = None
        self._topic_update: float = 0.0

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
            # NOTE: fewer files -- more overlap
            # yield hash_str[6:56]
            # NOTE: we ignore the last segment
            # yield hash_str[56:]

        all_segs = list(split_hash(message_hash.to_parseable()))
        segs = all_segs[:-1]
        rest = all_segs[-1]
        return os.path.join(self._path, *segs, f"{rest}{MSG_EXT}")

    def write_message(self, message: Message) -> MHash:
        mhash = message.get_hash()
        ensure_folder(os.path.dirname(self._compute_path(mhash)))
        with open_append(self._compute_path(mhash), text=True) as fout:
            fout.write(f"{self._escape(message.get_text())}\n")
        return message.get_hash()

    def _load_file(self, fname: str) -> Iterable[Message]:
        try:
            with open_read(fname, text=True) as fin:
                for line in fin:
                    line = line.rstrip()
                    if not line:
                        continue
                    text = self._unescape(line)
                    msg = Message(msg=text)
                    mhash = msg.get_hash()
                    self._cache.set(mhash, msg)
                    yield msg
        except FileNotFoundError:
            pass

    def read_message(self, message_hash: MHash) -> Message:
        res = self._cache.get(message_hash)
        if res is not None:
            return res
        for msg in self._load_file(self._compute_path(message_hash)):
            if msg.get_hash() == message_hash:
                res = msg
        if res is None:
            raise KeyError(f"no message for the hash: {message_hash}")
        return res

    def add_topic(self, topic: Message) -> MHash:
        if not topic.is_topic():
            raise ValueError(f"{topic}(\"{topic.get_text()}\") is not a topic")
        with open_append(self._topics, text=True) as fout:
            fout.write(f"{self._escape(topic.get_text())}\n")
        self._topic_cache = None
        return topic.get_hash()

    def get_topics(self, offset: int, limit: int | None) -> list[Message]:
        cur_time = time.monotonic()
        if (self._topic_cache is None
                or cur_time >= self._topic_update + RELOAD_TOPICS_FREQ):
            topic_cache = []
            try:
                with open_read(self._topics, text=True) as fin:
                    for line in fin:
                        line = line.rstrip()
                        if not line:
                            continue
                        text = self._unescape(line)
                        msg = Message(msg=text)
                        assert msg.is_topic()
                        topic_cache.append(msg)
            except FileNotFoundError:
                pass
            self._topic_cache = topic_cache
            self._topic_update = cur_time
        if limit is None:
            return self._topic_cache[offset:]
        return self._topic_cache[offset:offset + limit]

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        remain = count
        cur_path = self._path
        while remain > 0:
            candidates = list(get_folder(cur_path, MSG_EXT))
            if candidates:
                seg, recurse = candidates[rng.integers(0, len(candidates))]
                cur_path = os.path.join(cur_path, seg)
                if recurse:
                    continue
                cur = list(set((
                    msg.get_hash() for msg in self._load_file(cur_path))))
                if cur:
                    yield cur[rng.integers(0, len(cur))]
            remain -= 1
            cur_path = self._path

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:

        def get_level(
                cur_path: str,
                *,
                pbar: Callable[[], None] | None) -> Iterable[MHash]:
            for seg, recurse in get_folder(cur_path, MSG_EXT):
                full = os.path.join(cur_path, seg)
                if recurse:
                    yield from get_level(full, pbar=None)
                else:
                    yield from set(
                        msg.get_hash()
                        for msg in self._load_file(full))
                if pbar is not None:
                    pbar()

        if not progress_bar:
            yield from get_level(self._path, pbar=None)
        else:
            # FIXME: add stubs
            from tqdm.auto import tqdm  # type: ignore

            first_level_size = len(list(get_folder(self._path, MSG_EXT)))
            with tqdm(total=first_level_size) as pbar:
                yield from get_level(self._path, pbar=lambda: pbar.update(1))
