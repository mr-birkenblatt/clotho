import time
from typing import Dict, Generic, Optional, TypeVar


KT = TypeVar('KT')
VT = TypeVar('VT')


class LRU(Generic[KT, VT]):
    def __init__(
            self,
            max_items: int,
            soft_limit: Optional[int] = None) -> None:
        self._values: Dict[KT, VT] = {}
        self._times: Dict[KT, float] = {}
        self._max_items = max_items
        self._soft_limit = (
            max(1, int(max_items * 0.9)) if soft_limit is None else soft_limit)
        assert self._max_items >= self._soft_limit

    def get(self, key: KT) -> Optional[VT]:
        res = self._values.get(key)
        if res is not None:
            self._times[key] = time.monotonic()
        return res

    def set(self, key: KT, value: VT) -> None:
        self._values[key] = value
        self._times[key] = time.monotonic()
        self.gc()

    def gc(self) -> None:
        if len(self._values) <= self._max_items:
            return
        to_remove = sorted(
            self._times.items(),
            key=lambda item: item[1])[:-self._soft_limit]
        for rm_item in to_remove:
            key = rm_item[0]
            # FIXME: mypy bug?
            self._values.pop(key, None)  # type: ignore
            self._times.pop(key, None)
