import threading
import time
from typing import Callable, Generic, TYPE_CHECKING, TypeVar


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

    def trigger_update(self, key: KT, cur_time: float) -> None:
        self._dependent.trigger_update(self._convert(key), cur_time)

    def convert(self, key: KT) -> CT:
        return self._convert(key)


class EffectBase(Generic[KT]):
    def __init__(self) -> None:
        self._dependents: list[Dependent[KT, KeyType]] = []

    def add_dependent(self, dependent: Dependent[KT, KeyType]) -> None:
        self._dependents.append(dependent)

    def on_update(self, key: KT) -> None:
        cur_time = time.monotonic()
        for dependent in self._dependents:
            dependent.trigger_update(key, cur_time)

    def settle_all(self) -> int:
        raise NotImplementedError()


class EffectRoot(Generic[KT, VT], EffectBase[KT]):
    def settle_all(self) -> int:
        return 0

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

    def update_value(self, key: KT, value: VT) -> VT | None:
        res = self.do_update_value(key, value)
        self.on_update(key)
        return res

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def set_value(self, key: KT, value: VT) -> None:
        self.do_set_value(key, value)
        self.on_update(key)

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def set_new_value(self, key: KT, value: VT) -> bool:
        was_set = self.do_set_new_value(key, value)
        if was_set:
            self.on_update(key)
        return was_set


class SetRootType(Generic[KT, VT], EffectRoot[KT, set[VT]]):
    def do_add_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def add_value(self, key: KT, value: VT) -> bool:
        res = self.do_add_value(key, value)
        self.on_update(key)
        return res

    def do_remove_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def remove_value(self, key: KT, value: VT) -> bool:
        res = self.do_remove_value(key, value)
        self.on_update(key)
        return res


class EffectDependent(Generic[KT, VT], EffectBase[KT]):
    def __init__(
            self,
            *,
            parents: tuple[EffectBase[PT], ...],
            convert: Callable[[PT], KT],
            effect: Callable[[KT], None],
            delay: float) -> None:
        super().__init__()
        self._pending: dict[KT, float] = {}
        self._parents = parents
        self._effect = effect
        self._delay = delay
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
        end_time = self._delay + cur_time
        if prev_time is None or prev_time > end_time:
            self._pending[key] = end_time
        self.init_thread()

    def poll_update(self, cur_time: float) -> float | None:
        next_time: float | None = None
        for (pkey, update_time) in list(self._pending.items()):
            if update_time > cur_time:
                if next_time is None or update_time < next_time:
                    next_time = update_time
                continue
            pkval = self._pending.pop(pkey, None)
            if pkval is not None:
                success = False
                try:
                    self.execute_update(pkey)
                    success = True
                finally:
                    if not success:
                        self._pending[pkey] = pkval
        return next_time

    def settle_all(self) -> int:
        count = 0
        for parent in self._parents:
            count += parent.settle_all()
        for pkey in list(self._pending.keys()):
            pkval = self._pending.pop(pkey, None)
            if pkval is not None:
                success = False
                try:
                    self.execute_update(pkey)
                    count += 1
                    success = True
                finally:
                    if not success:
                        self._pending[pkey] = pkval
        return count

    def execute_update(self, key: KT) -> None:
        self._effect(key)

    def retrieve_value(self, key: KT) -> VT | None:
        raise NotImplementedError()

    def get_value(self, key: KT, default: VT) -> VT:
        res = self.maybe_get_value(key)
        if res is None:
            return default
        return res

    def maybe_get_value(self, key: KT) -> VT | None:
        return self.retrieve_value(key)

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def set_value(self, key: KT, value: VT) -> None:
        self.do_set_value(key, value)
        self.on_update(key)

    def do_update_value(self, key: KT, value: VT) -> VT | None:
        raise NotImplementedError()

    def update_value(self, key: KT, value: VT) -> VT | None:
        res = self.do_update_value(key, value)
        self.on_update(key)
        return res

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        raise NotImplementedError()

    def set_new_value(self, key: KT, value: VT) -> bool:
        was_set = self.do_set_new_value(key, value)
        if was_set:
            self.on_update(key)
        return was_set
