import contextlib
import threading
from typing import Callable, Generic, Iterator, Literal, TYPE_CHECKING, TypeVar

import pandas as pd

from misc.util import now_ts


if TYPE_CHECKING:
    from typing import Any


Freshness = Literal["old", "needs_update", "current"]


class EqType:
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        raise NotImplementedError()


BaseKeyType = int | str | EqType
KeyType = tuple[BaseKeyType, ...] | BaseKeyType


KT = TypeVar('KT', bound=KeyType)
VT = TypeVar('VT')
PT = TypeVar('PT', bound=KeyType)
CT = TypeVar('CT', bound=KeyType)
AT = TypeVar('AT')


OLD_THRESHOLD = pd.Timedelta(seconds=60.0 * 60.0)


def set_old_threshold(old_threshold: float) -> None:
    global OLD_THRESHOLD

    OLD_THRESHOLD = pd.Timedelta(seconds=old_threshold)


def get_old_threshold() -> float:
    return OLD_THRESHOLD / pd.Timedelta("1s")


class Dependent(Generic[KT, CT]):
    def __init__(
            self,
            dependent: 'EffectDependent[CT, Any]',
            convert: Callable[[KT], CT]) -> None:
        self._dependent = dependent
        self._convert = convert

    def set_outdated(self, key: KT, now: pd.Timestamp | None) -> None:
        self._dependent.set_outdated(self.convert(key), now)

    def convert(self, key: KT) -> CT:
        return self._convert(key)


class EffectBase(Generic[KT]):
    def __init__(self) -> None:
        self._dependents: list[Dependent[KT, KeyType]] = []

    def add_dependent(self, dependent: Dependent[KT, KeyType]) -> None:
        self._dependents.append(dependent)

    def on_update(self, key: KT, now: pd.Timestamp | None) -> None:
        for dependent in self._dependents:
            dependent.set_outdated(key, now)


class EffectRoot(Generic[KT, VT], EffectBase[KT]):
    def get_value(self, key: KT, default: VT) -> VT:
        res = self.maybe_get_value(key)
        if res is None:
            return default
        return res

    def maybe_get_value(self, key: KT) -> VT | None:
        raise NotImplementedError()


class ValueRootType(Generic[KT, VT], EffectRoot[KT, VT]):
    def do_update_value(self, key: KT, value: VT) -> VT | None:
        raise NotImplementedError()

    def update_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> VT | None:
        res = self.do_update_value(key, value)
        self.on_update(key, now)
        return res

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def set_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> None:
        self.do_set_value(key, value)
        self.on_update(key, now)

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def set_new_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> bool:
        was_set = self.do_set_new_value(key, value)
        if was_set:
            self.on_update(key, now)
        return was_set


class SetRootType(Generic[KT, VT], EffectRoot[KT, set[VT]]):
    def do_add_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def add_value(self, key: KT, value: VT, now: pd.Timestamp | None) -> bool:
        res = self.do_add_value(key, value)
        self.on_update(key, now)
        return res

    def do_remove_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def remove_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> bool:
        res = self.do_remove_value(key, value)
        self.on_update(key, now)
        return res

    def has_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def get_size(self, key: KT) -> int:
        raise NotImplementedError()


class EffectDependent(Generic[KT, VT], EffectBase[KT]):
    def __init__(
            self,
            *,
            parents: tuple[EffectBase[PT], ...],
            convert: Callable[[PT], KT],
            effect: Callable[[KT, pd.Timestamp | None], None]) -> None:
        super().__init__()
        self._pending: dict[KT, float] = {}
        self._parents = parents
        self._effect = effect
        self._thread: threading.Thread | None = None
        self._update_lock = threading.RLock()
        self._is_updating = False
        for parent in self._parents:
            parent.add_dependent(Dependent(self, convert))  # type: ignore

    def init_thread(self) -> None:
        if self._thread is None:
            th = threading.Thread(target=self.updater, daemon=True)
            self._thread = th
            th.start()

    def set_outdated(self, key: KT, now: pd.Timestamp | None) -> None:
        self.on_set_outdated(key, now_ts() if now is None else now)
        self.on_update(key, now)

    def updater(self) -> None:
        try:
            while True:
                if self._thread is not threading.current_thread():
                    break
                next_task = self.pending_outdated()
                if next_task is None:
                    break
                key, when = next_task
                now = now_ts()
                self.execute_update(key, max(now, when))
        finally:
            self._thread = None

    @contextlib.contextmanager
    def no_deferred_updates(self) -> Iterator[None]:
        with self._update_lock:
            yield

    def execute_update(self, key: KT, now: pd.Timestamp | None) -> None:
        with self._update_lock:
            if self._is_updating:
                raise RuntimeError(
                    f"attempt to update while updating {self} {key} {now}")
            self._is_updating = True
            try:
                self._effect(key, now)
                self.clear_outdated(key, now)
            finally:
                self._is_updating = False

    def get_value(self, key: KT, default: VT, when: pd.Timestamp | None) -> VT:
        res = self.maybe_get_value(key, when)
        if res is None:
            return default
        return res

    def maybe_get_value(
            self,
            key: KT,
            when: pd.Timestamp | None) -> VT | None:
        value, outdated_ts = self.retrieve_value(key)
        if self._is_updating:
            return value
        is_outdated = self._is_outdated(outdated_ts, when)
        if is_outdated != "old" and value is not None:
            self.maybe_compute(key, outdated_ts)
            return value
        self.execute_update(key, when)
        value, _ = self.retrieve_value(key)
        return value

    def _is_outdated(
            self,
            outdated_ts: pd.Timestamp | None,
            when: pd.Timestamp | None) -> Freshness:
        if outdated_ts is None:
            return "current"
        if when is None:
            return "old"
        if when < outdated_ts:
            return "current"
        if when > outdated_ts + OLD_THRESHOLD:
            return "old"
        return "needs_update"

    def maybe_compute(self, key: KT, marker: pd.Timestamp | None) -> None:
        if marker is not None:
            self.request_compute(key)

    def request_compute(self, key: KT) -> None:
        self.on_request_compute(key)
        self.init_thread()

    def set_value(self, key: KT, value: VT, now: pd.Timestamp | None) -> None:
        self.do_set_value(key, value, now)
        self.on_update(key, now)

    def update_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> VT | None:
        res = self.do_update_value(key, value, now)
        self.on_update(key, now)
        return res

    def set_new_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> bool:
        was_set = self.do_set_new_value(key, value, now)
        if was_set:
            self.on_update(key, now)
        return was_set

    def pending_outdated(self) -> tuple[KT, pd.Timestamp] | None:
        raise NotImplementedError()

    def on_request_compute(self, key: KT) -> None:
        raise NotImplementedError()

    def on_set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        raise NotImplementedError()

    def clear_outdated(self, key: KT, now: pd.Timestamp | None) -> None:
        raise NotImplementedError()

    def retrieve_value(self, key: KT) -> tuple[VT | None, pd.Timestamp | None]:
        raise NotImplementedError()

    def do_set_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> None:
        raise NotImplementedError()

    def do_update_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> VT | None:
        raise NotImplementedError()

    def do_set_new_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> bool:
        raise NotImplementedError()


class ListEffectDependent(Generic[KT, VT], EffectDependent[KT, list[VT]]):
    def do_get_value_range(
            self,
            key: KT,
            from_ix: int | None,
            to_ix: int | None) -> tuple[list[VT] | None, pd.Timestamp | None]:
        raise NotImplementedError()

    def get_value_range(
            self,
            key: KT,
            from_ix: int | None,
            to_ix: int | None,
            when: pd.Timestamp | None) -> list[VT]:
        value, outdated_ts = self.do_get_value_range(key, from_ix, to_ix)
        if self._is_updating:
            return [] if value is None else value
        is_outdated = self._is_outdated(outdated_ts, when)
        if is_outdated != "old" and value is not None:
            self.maybe_compute(key, outdated_ts)
            return value
        self.execute_update(key, when)
        value, _ = self.do_get_value_range(key, from_ix, to_ix)
        return [] if value is None else value

    def __getitem__(
            self, arg: tuple[KT, slice, pd.Timestamp | None]) -> list[VT]:
        key, slicer, now = arg
        if slicer.step is None or slicer.step > 0:
            res = self.get_value_range(key, slicer.start, slicer.stop, now)
        else:
            flip_start = slicer.stop
            if flip_start is not None:
                if flip_start != -1:
                    flip_start += 1
                else:
                    return []
            flip_stop = slicer.start
            if flip_stop is not None:
                if flip_stop != -1:
                    flip_stop += 1
                else:
                    flip_stop = None
            res = self.get_value_range(key, flip_start, flip_stop, now)
        if slicer.step is not None and slicer.step != 1:
            return res[::slicer.step]
        return res

    def do_get_size(self, key: KT) -> tuple[int | None, pd.Timestamp | None]:
        value, outdated_ts = self.do_get_value_range(key, 0, None)
        return None if value is None else len(value), outdated_ts

    def get_size(self, key: KT, when: pd.Timestamp | None) -> int:
        size, outdated_ts = self.do_get_size(key)
        if self._is_updating:
            return 0 if size is None else size
        is_outdated = self._is_outdated(outdated_ts, when)
        if is_outdated != "old" and size is not None:
            self.maybe_compute(key, outdated_ts)
            return size
        self.execute_update(key, when)
        size, _ = self.do_get_size(key)
        return 0 if size is None else size
