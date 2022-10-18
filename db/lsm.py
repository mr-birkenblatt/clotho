import atexit
import time
import uuid
from typing import Iterable

from misc.lru import LRU


class KeyRange:
    def __init__(self, shortest: str, longest: str | None) -> None:
        if longest is not None:
            if len(shortest) > len(longest):
                raise ValueError(f"{shortest} > {longest}")
            if not longest.startswith(shortest):
                raise ValueError(f"{shortest} |> {longest}")
        self._shortest = shortest
        self._longest = longest

    def match(self, key: str) -> bool:
        if not key.startswith(self._shortest):
            return False
        # FIXME: define longest
        return True

    def is_beginning(self, key: str) -> bool:
        return self._shortest.startswith(key)


class MissingKey:  # pylint: disable=too-few-public-methods
    pass


MISSING_KEY = MissingKey()


VType = MissingKey | str | int | float | list[str | int | float]


def is_missing_key(value: VType) -> bool:
    return value is MISSING_KEY or isinstance(value, MissingKey)


class CacheCoordinator:
    def invalidate_cache(self, pid: str, key_range: KeyRange) -> None:
        raise NotImplementedError()

    def clear_cache(self, lsm: 'LSM', pid: str, key_range: KeyRange) -> None:
        lsm.clear_cache(pid, key_range)


class ChunkCoordinator:
    def write_values(
            self,
            cur_time: float,
            values: dict[str, VType],
            lsm: 'LSM') -> None:
        raise NotImplementedError()

    def fetch_values(
            self, key: str, lsm: 'LSM') -> Iterable[tuple[str, VType]]:
        raise NotImplementedError()

    def clear_cache(self, key_range: KeyRange) -> None:
        raise NotImplementedError()


class LSM:
    def __init__(
            self,
            *,
            cache_coordinator: CacheCoordinator,
            cache_size: int = 100000,
            write_cache_freq: float = 5 * 60,
            write_cache_size: int = 10000,
            chunk_coordinator: ChunkCoordinator) -> None:
        self._write_cache: dict[str, VType] = {}
        self._cache: LRU[str, VType] = LRU(cache_size)
        self._pid = uuid.uuid4().hex
        self._cache_coordinator = cache_coordinator
        self._last_write = time.monotonic()
        self._write_cache_freq = write_cache_freq
        self._write_cache_size = write_cache_size
        self._chunk_coordinator = chunk_coordinator
        atexit.register(self.flushall)

    def invalidate_cache(self, key_range: KeyRange) -> None:
        self._cache_coordinator.invalidate_cache(self._pid, key_range)

    def clear_cache(self, pid: str, key_range: KeyRange) -> None:
        if pid == self._pid:
            return
        self._cache.clear_keys(key_range.match)

    def maybe_flush(self) -> None:
        if self._last_write + self._write_cache_freq < time.monotonic():
            self.flushall()
        elif len(self._write_cache) > self._write_cache_size:
            self.flushall()

    def flushall(self) -> None:
        cur_time = time.time()
        write_cache = self._write_cache
        self._write_cache = {}
        self._last_write = cur_time
        self._chunk_coordinator.write_values(cur_time, write_cache, self)

    def _fetch(self, key: str) -> VType:
        res: VType = MISSING_KEY
        for cur in self._chunk_coordinator.fetch_values(key, self):
            cur_key, cur_value = cur
            if cur_key == key:
                res = cur_value
            self._cache.set(cur_key, cur_value)
        if is_missing_key(res):
            self._cache.set(key, res)
        return res

    def put(self, key: str, value: VType) -> None:
        self._write_cache[key] = value
        self.maybe_flush()

    def remove(self, key: str) -> None:
        self.put(key, MISSING_KEY)

    def get(
            self,
            key: str,
            default: VType | None = None) -> VType | None:
        res = self._write_cache.get(key, None)

        def prepare(res: VType) -> VType | None:
            return default if is_missing_key(res) else res

        if res is not None:
            return prepare(res)
        res = self._cache.get(key)
        if res is not None:
            return prepare(res)
        return prepare(self._fetch(key))
