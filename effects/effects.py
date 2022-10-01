import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)


KT = TypeVar('KT')
VT = TypeVar('VT')
LT = TypeVar('LT', bound=Tuple['EffectBase', ...])


DELAY = 5


class EffectBase(Generic[KT]):
    def __init__(self) -> None:
        self._dependents: List['EffectDependent[KT, Any]'] = []

    def add_dependent(
            self, dependent: 'EffectDependent[KT, Any]') -> None:
        self._dependents.append(dependent)

    def on_update(self, key: KT) -> None:
        cur_time = time.monotonic()
        for dependent in self._dependents:
            dependent.trigger_update(key, cur_time)


class EffectRoot(Generic[KT, VT], EffectBase[KT]):
    def get_value(self, key: KT) -> Optional[VT]:
        raise NotImplementedError()


class ValueRootType(Generic[KT, VT], EffectRoot[KT, VT]):
    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def set_value(self, key: KT, value: VT) -> None:
        self.do_set_value(key, value)
        self.on_update(key)


class SetRootType(Generic[KT, VT], EffectRoot[KT, Set[VT]]):
    def do_add_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def add_value(self, key: KT, value: VT) -> None:
        self.do_add_value(key, value)
        self.on_update(key)

    def do_remove_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def remove_value(self, key: KT, value: VT) -> None:
        self.do_remove_value(key, value)
        self.on_update(key)


class EffectDependent(Generic[KT, VT], EffectBase[KT]):
    def __init__(
            self,
            parents: LT,
            effect: Callable[
                ['EffectDependent[KT, VT]', LT, KT], None]) -> None:
        super().__init__()
        self._pending: Dict[KT, float] = {}
        self._parents = parents
        self._effect = effect
        self._thread: Optional[threading.Thread] = None
        for parent in self._parents:
            parent.add_dependent(self)

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
                cur_time = time.monotonic()
                next_time = self.poll_update(cur_time)
                if next_time is None:
                    break
                pause = next_time - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
        finally:
            self._thread = None

    def trigger_update(self, key: KT, cur_time: float) -> None:
        prev_time = self._pending.get(key)
        if prev_time is None or prev_time > DELAY + cur_time:
            self._pending[key] = DELAY + cur_time
        self.init_thread()

    def poll_update(self, cur_time: float) -> Optional[float]:
        next_time: Optional[float] = None
        to_update: List[KT] = []
        for (key, update_time) in list(self._pending.items()):
            if update_time > cur_time:
                if next_time is None or update_time < next_time:
                    next_time = update_time
                continue
            self._pending.pop(key, default=None)  # type: ignore
            to_update.append(key)
        for key in to_update:
            self.execute_update(key)
        return next_time

    def execute_update(self, key: KT) -> None:
        self._effect(self, self._parents, key)

    def retrieve_value(self, key: KT) -> Optional[VT]:
        raise NotImplementedError()

    def get_value(self, key: KT) -> Optional[VT]:
        res = self.retrieve_value(key)
        if res is not None:
            return res
        self.execute_update(key)
        return self.retrieve_value(key)


class ValueDependentType(Generic[KT, VT], EffectDependent[KT, VT]):
    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def set_value(self, key: KT, value: VT) -> None:
        self.do_set_value(key, value)
        self.on_update(key)


class SetDependentType(Generic[KT, VT], EffectDependent[KT, Set[VT]]):
    def do_add_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def add_value(self, key: KT, value: VT) -> None:
        self.do_add_value(key, value)
        self.on_update(key)

    def do_remove_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def remove_value(self, key: KT, value: VT) -> None:
        self.do_remove_value(key, value)
        self.on_update(key)
