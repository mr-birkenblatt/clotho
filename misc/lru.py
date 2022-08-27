import time
from typing import Dict, Generic, Optional, TypeVar


KT = TypeVar('KT')
VT = TypeVar('VT')


class LRU(Generic[KT, VT]):
    def __init__(self, max_items: int) -> None:
        self._values: Dict[KT, VT] = {}
        self._times: Dict[KT, float] = {}
        self._max_items = max_items

    def get(self, key: KT) -> Optional[VT]:
        # FIXME: mypy bug?
        res = self._values.get(key, None)  # type: ignore
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
            key=lambda item: item[1])[:-self._max_items]
        for rm_item in to_remove:
            key = rm_item[0]
            # FIXME: mypy bug?
            self._values.pop(key, None)  # type: ignore
            self._times.pop(key, None)
