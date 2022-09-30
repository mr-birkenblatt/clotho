import atexit
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple, Union

from misc.lru import LRU


class MissingKey:  # pylint: disable=too-few-public-methods
    pass


MISSING_KEY = MissingKey()


VType = Union[MissingKey, str, int, float, List[Union[str, int, float]]]


def is_missing_key(value: VType) -> bool:
    return value is MISSING_KEY or isinstance(value, MissingKey)


class CacheCoordinator:
    def invalidate_cache(self, pid: str, prefix: str) -> None:
        raise NotImplementedError()

    def clear_cache(self, lsm: 'LSM', pid: str, prefix: str) -> None:
        lsm.clear_cache(pid, prefix)


class ChunkCoordinator:
    def write_values(
            self,
            cur_time: float,
            values: Dict[str, VType],
            lsm: 'LSM') -> None:
        raise NotImplementedError()

    def fetch_values(
            self, key: str, lsm: 'LSM') -> Iterable[Tuple[str, VType]]:
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
        self._write_cache: Dict[str, VType] = {}
        self._cache: LRU[str, VType] = LRU(cache_size)
        self._pid = uuid.uuid4().hex
        self._cache_coordinator = cache_coordinator
        self._last_write = time.monotonic()
        self._write_cache_freq = write_cache_freq
        self._write_cache_size = write_cache_size
        self._chunk_coordinator = chunk_coordinator
        atexit.register(self.flushall)

    def invalidate_cache(self, prefix: str) -> None:
        self._cache_coordinator.invalidate_cache(self._pid, prefix)

    def clear_cache(self, pid: str, prefix: str) -> None:
        if pid == self._pid:
            return
        self._cache.clear_keys(lambda key: key.startswith(prefix))

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
            default: Optional[VType] = None) -> Optional[VType]:
        res = self._write_cache.get(key, None)

        def prepare(res: VType) -> Optional[VType]:
            return default if is_missing_key(res) else res

        if res is not None:
            return prepare(res)
        res = self._cache.get(key)
        if res is not None:
            return prepare(res)
        return prepare(self._fetch(key))
