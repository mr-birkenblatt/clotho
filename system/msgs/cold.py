import gzip
import os
import threading
import time
from typing import Callable, Iterable, TextIO

import numpy as np

from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


class ColdStore(MessageStore):
    def __init__(self, msgs_root: str, keep_alive: float) -> None:
        super().__init__()
        self._msg_file = os.path.join(msgs_root, "msgs.zip")
        self._topic_file = os.path.join(msgs_root, "topics.zip")
        self._mout: TextIO | None = None
        self._tout: TextIO | None = None
        cur = time.monotonic()
        self._mtimeout = cur
        self._ttimeout = cur
        self._lock = threading.RLock()
        self._keep_alive = keep_alive
        self._buff: LRU[MHash, Message] = LRU(100)
        self._th_error: BaseException | None = None

    def _write_io(
            self,
            get_out: Callable[[], TextIO | None],
            set_out: Callable[[TextIO | None], None],
            get_file: Callable[[], str],
            get_timeout: Callable[[], float],
            set_timeout: Callable[[float], None],
            text: str) -> None:

        def writer(cond: threading.Condition) -> None:
            try:
                out = None
                with self._lock:
                    out = gzip.open(get_file(), mode="at", encoding="utf-8")
                    set_out(out)
                    cond.notify_all()
                while True:
                    cur = time.monotonic()
                    timeout = get_timeout()
                    if cur >= timeout:
                        break
                    time.sleep(timeout - cur)
            except BaseException as e:  # pylint: disable=broad-except
                self._th_error = e
            finally:
                if out is not None:
                    with self._lock:
                        out.flush()
                        out.close()
                        if out is get_out():
                            set_out(None)

        if self._th_error is not None:
            raise self._th_error
        with self._lock:
            set_timeout(time.monotonic() + self._keep_alive)
            out = get_out()
            if out is not None and not out.closed:
                out.write(text)
                return
            set_out(None)
            cond = threading.Condition(self._lock)
            th = threading.Thread(target=writer, args=(cond,))
            th.start()
        with self._lock:
            while get_out() is None:
                if self._th_error is not None:
                    raise self._th_error
                cond.wait(1.0)
        self._write_io(
            get_out, set_out, get_file, get_timeout, set_timeout, text)

    def _write_message_io(self, text: str) -> None:

        def get_out() -> TextIO | None:
            return self._mout

        def set_out(out: TextIO | None) -> None:
            self._mout = out

        def get_file() -> str:
            return self._msg_file

        def get_timeout() -> float:
            return self._mtimeout

        def set_timeout(timeout: float) -> None:
            self._mtimeout = timeout

        self._write_io(
            get_out, set_out, get_file, get_timeout, set_timeout, text)

    def _write_topic_io(self, text: str) -> None:

        def get_out() -> TextIO | None:
            return self._tout

        def set_out(out: TextIO | None) -> None:
            self._tout = out

        def get_file() -> str:
            return self._topic_file

        def get_timeout() -> float:
            return self._ttimeout

        def set_timeout(timeout: float) -> None:
            self._ttimeout = timeout

        self._write_io(
            get_out, set_out, get_file, get_timeout, set_timeout, text)

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n")

    @staticmethod
    def _unescape(text: str) -> str:
        return text.replace("\\n", "\n").replace("\\\\", "\\")

    def write_message(self, message: Message) -> MHash:
        self._write_message_io(f"{self._escape(message.get_text())}\n")
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
        self._write_topic_io(f"{self._escape(topic.get_text())}\n")
        return topic.get_hash()

    def _get_topics(self) -> Iterable[Message]:
        with gzip.open(self._topic_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                if line.endswith("\n"):
                    line = line[:-1]
                yield Message(msg=self._unescape(line))

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
        with gzip.open(self._msg_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                if line.endswith("\n"):
                    line = line[:-1]
                msg = Message(msg=self._unescape(line))
                res = msg.get_hash()
                self._buff.set(res, msg)
                yield res
