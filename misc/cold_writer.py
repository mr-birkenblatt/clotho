import gzip
import threading
import time
from typing import Iterable, TextIO

from misc.util import escape, unescape


class ColdAccess:
    def __init__(self, fname: str, *, keep_alive: float) -> None:
        self._fname = fname
        self._out: TextIO | None = None
        cur = time.monotonic()
        self._timeout = cur
        self._lock = threading.RLock()
        self._keep_alive = keep_alive
        self._th_error: BaseException | None = None

    def _write_line(self, text: str) -> None:

        def writer(cond: threading.Condition) -> None:
            try:
                out = None
                with self._lock:
                    out = gzip.open(self._fname, mode="at", encoding="utf-8")
                    self._out = out
                    cond.notify_all()
                while True:
                    cur = time.monotonic()
                    timeout = self._timeout
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
                        if out is self._out:
                            self._out = None

        if self._th_error is not None:
            raise self._th_error
        with self._lock:
            self._timeout = time.monotonic() + self._keep_alive
            out = self._out
            if out is not None and not out.closed:
                out.write(text)
                return
            self._out = None
            cond = threading.Condition(self._lock)
            th = threading.Thread(target=writer, args=(cond,))
            th.start()
        with self._lock:
            while self._out is None:
                if self._th_error is not None:
                    raise self._th_error
                cond.wait(1.0)
        self._write_line(text)

    @staticmethod
    def _escape(text: str) -> str:
        return escape(text, {"\n": "n"})

    @staticmethod
    def _unescape(text: str) -> str:
        return unescape(text, {"n": "\n"})

    def write_line(self, line: str) -> None:
        self._write_line(f"{self._escape(line)}\n")

    def enumerate_lines(self) -> Iterable[str]:
        with gzip.open(self._fname, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.removesuffix("\n")
                yield self._unescape(line)
