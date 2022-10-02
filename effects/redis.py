from typing import Callable, Generic, Set, Tuple, TypeVar

from effects.effects import (
    EffectBase,
    EffectDependent,
    SetDependentType,
    SetRootType,
    ValueDependentType,
    ValueRootType,
)
from misc.redis import RedisConnection, RedisModule


KT = TypeVar('KT')
VT = TypeVar('VT')
LT = TypeVar('LT', bound=Tuple[EffectBase, ...])


class ValueRootRedisType(Generic[KT, VT], ValueRootType[KT, VT]):
    def __init__(self, module: RedisModule) -> None:
        super().__init__()
        self._redis = RedisConnection(module)

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()


class ListRootRedisType(Generic[KT, VT], ValueRootType[KT, VT]):
    def __init__(self, module: RedisModule) -> None:
        super().__init__()
        self._redis = RedisConnection(module)

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()


class SetRootRedisType(Generic[KT, VT], SetRootType[KT, VT]):
    def __init__(self, module: RedisModule) -> None:
        super().__init__()
        self._redis = RedisConnection(module)

    def do_add_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def do_remove_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()


class ValueDependentRedisType(Generic[KT, VT], ValueDependentType[KT, VT]):
    def __init__(
            self,
            module: RedisModule,
            parents: LT,
            effect: Callable[[EffectDependent[KT, VT], LT, KT], None]) -> None:
        super().__init__(parents, effect)
        self._redis = RedisConnection(module)

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()


class ListDependentRedisType(Generic[KT, VT], ValueDependentType[KT, VT]):
    def __init__(
            self,
            module: RedisModule,
            parents: LT,
            effect: Callable[[EffectDependent[KT, VT], LT, KT], None]) -> None:
        super().__init__(parents, effect)
        self._redis = RedisConnection(module)

    def do_set_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()


class SetDependentRedisType(Generic[KT, VT], SetDependentType[KT, VT]):
    def __init__(
            self,
            module: RedisModule,
            parents: LT,
            effect: Callable[[EffectDependent[KT, VT], LT, KT], None]) -> None:
        super().__init__(parents, effect)
        self._redis = RedisConnection(module)

    def do_add_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()

    def do_remove_value(self, key: KT, value: VT) -> None:
        raise NotImplementedError()
