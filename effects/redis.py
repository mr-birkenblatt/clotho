from typing import Callable, Generic, Optional, Set, Tuple, TypeVar

from effects.effects import (
    EffectBase,
    EffectDependent,
    SetDependentType,
    SetRootType,
    ValueDependentType,
    ValueRootType,
)
from misc.redis import RedisConnection, RedisModule
from misc.util import json_compact, json_read


KT = TypeVar('KT')
VT = TypeVar('VT')
LT = TypeVar('LT', bound=Tuple[EffectBase, ...])


class StrRootRedisType(Generic[KT, VT], ValueRootType[KT, VT]):
    def __init__(
            self, module: RedisModule, key_fn: Callable[[KT], VT]) -> None:
        super().__init__()
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_set_value(self, key: KT, value: VT) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.set(rkey, json_compact(value))

    def get_value(self, key: KT) -> Optional[VT]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.get(rkey)
        return json_read(res) if res is not None else None


class SetRootRedisType(Generic[KT], SetRootType[KT, str]):
    def __init__(
            self, module: RedisModule, key_fn: Callable[[KT], str]) -> None:
        super().__init__()
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_add_value(self, key: KT, value: str) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.sadd(rkey, value.encode("utf-8"))

    def do_remove_value(self, key: KT, value: str) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.sremove(rkey, value.encode("utf-8"))

    def get_value(self, key: KT) -> Optional[Set[str]]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            return set(mem.decode("utf-8") for mem in conn.smembers(rkey))


class ValueDependentRedisType(Generic[KT, VT], ValueDependentType[KT, VT]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            parents: LT,
            effect: Callable[[EffectDependent[KT, VT], LT, KT], None]) -> None:
        super().__init__(parents, effect)
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_set_value(self, key: KT, value: VT) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.set(rkey, json_compact(value))

    def retrieve_value(self, key: KT) -> Optional[VT]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.get(rkey)
        return json_read(res) if res is not None else None


class SetDependentRedisType(Generic[KT], SetDependentType[KT, str]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            parents: LT,
            effect: Callable[[EffectDependent[KT, Set[str]], LT, KT], None],
            ) -> None:
        super().__init__(parents, effect)
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_add_value(self, key: KT, value: str) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.sadd(rkey, value.encode("utf-8"))

    def do_remove_value(self, key: KT, value: str) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.sremove(rkey, value.encode("utf-8"))

    def retrieve_value(self, key: KT) -> Optional[Set[str]]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.get(rkey)
        return json_read(res) if res is not None else None
