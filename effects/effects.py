import threading
from typing import Callable, Generic, TYPE_CHECKING, TypeVar

import pandas as pd

from misc.util import now_ts


if TYPE_CHECKING:
    from typing import Any


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


class Dependent(Generic[KT, CT]):
    def __init__(
            self,
            dependent: 'EffectDependent[CT, Any]',
            convert: Callable[[KT], CT]) -> None:
        self._dependent = dependent
        self._convert = convert

    def set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        self._dependent.set_outdated(self._convert(key), now)

    def convert(self, key: KT) -> CT:
        return self._convert(key)


class EffectBase(Generic[KT]):
    def __init__(self) -> None:
        self._dependents: list[Dependent[KT, KeyType]] = []

    def add_dependent(self, dependent: Dependent[KT, KeyType]) -> None:
        self._dependents.append(dependent)

    def on_update(self, key: KT, now: pd.Timestamp) -> None:
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

    def update_value(self, key: KT, value: VT, now: pd.Timestamp) -> VT | None:
        res = self.do_update_value(key, value)
        self.on_update(key, now)
        return res

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def set_value(self, key: KT, value: VT, now: pd.Timestamp) -> None:
        self.do_set_value(key, value)
        self.on_update(key, now)

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def set_new_value(self, key: KT, value: VT, now: pd.Timestamp) -> bool:
        was_set = self.do_set_new_value(key, value)
        if was_set:
            self.on_update(key, now)
        return was_set


class SetRootType(Generic[KT, VT], EffectRoot[KT, set[VT]]):
    def do_add_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def add_value(self, key: KT, value: VT, now: pd.Timestamp) -> bool:
        res = self.do_add_value(key, value)
        self.on_update(key, now)
        return res

    def do_remove_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def remove_value(self, key: KT, value: VT, now: pd.Timestamp) -> bool:
        res = self.do_remove_value(key, value)
        self.on_update(key, now)
        return res

    def has_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()


class EffectDependent(Generic[KT, VT], EffectBase[KT]):
    def __init__(
            self,
            *,
            parents: tuple[EffectBase[PT], ...],
            convert: Callable[[PT], KT],
            effect: Callable[[KT], None]) -> None:
        super().__init__()
        self._pending: dict[KT, float] = {}
        self._parents = parents
        self._effect = effect
        self._thread: threading.Thread | None = None
        for parent in self._parents:
            parent.add_dependent(Dependent(self, convert))  # type: ignore

    def init_thread(self) -> None:
        if self._thread is None:
            th = threading.Thread(target=self.updater, daemon=True)
            self._thread = th
            th.start()

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

    def execute_update(self, key: KT, now: pd.Timestamp) -> None:
        self._effect(key)
        self.clear_outdated(key, now)

    def get_value(self, key: KT, default: VT) -> VT:
        res = self.maybe_get_value(key)
        if res is None:
            return default
        return res

    def maybe_get_value(
            self,
            key: KT) -> VT | None:
        return self.retrieve_value(key)[0]

    def poll_value(
            self,
            key: KT,
            when: pd.Timestamp | None) -> tuple[VT | None, bool]:
        value, outdated_ts = self.retrieve_value(key)
        return value, self._is_outdated(outdated_ts, when)

    def is_outdated(self, key: KT, when: pd.Timestamp | None) -> bool:
        return self.poll_value(key, when)[1]

    def _is_outdated(
            self,
            outdated_ts: pd.Timestamp | None,
            when: pd.Timestamp | None) -> bool:
        if outdated_ts is None:
            return False
        if when is None:
            return True
        return when >= outdated_ts

    def maybe_compute(self, key: KT, marker: pd.Timestamp | None) -> None:
        if marker is not None:
            self.request_compute(key)

    def request_compute(self, key: KT) -> None:
        self.on_request_compute(key)
        self.init_thread()

    def set_value(self, key: KT, value: VT, now: pd.Timestamp) -> None:
        self.do_set_value(key, value)
        self.on_update(key, now)

    def update_value(self, key: KT, value: VT, now: pd.Timestamp) -> VT | None:
        res = self.do_update_value(key, value)
        self.on_update(key, now)
        return res

    def set_new_value(self, key: KT, value: VT, now: pd.Timestamp) -> bool:
        was_set = self.do_set_new_value(key, value)
        if was_set:
            self.on_update(key, now)
        return was_set

    def pending_outdated(self) -> tuple[KT, pd.Timestamp] | None:
        raise NotImplementedError()

    def on_request_compute(self, key: KT) -> None:
        raise NotImplementedError()

    def set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        raise NotImplementedError()

    def clear_outdated(self, key: KT, now: pd.Timestamp) -> None:
        raise NotImplementedError()

    def retrieve_value(self, key: KT) -> tuple[VT | None, pd.Timestamp | None]:
        raise NotImplementedError()

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def do_update_value(self, key: KT, value: VT) -> VT | None:
        raise NotImplementedError()

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()
