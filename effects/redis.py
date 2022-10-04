from typing import Callable, Generic, Iterable, Optional, Set, Tuple, TypeVar

from effects.effects import (
    EffectBase,
    EffectDependent,
    SetRootType,
    ValueRootType,
)
from misc.redis import RedisConnection, RedisModule
from misc.util import json_compact, json_read


KT = TypeVar('KT')
VT = TypeVar('VT')
LT = TypeVar('LT', bound=Tuple[EffectBase, ...])


class ValueRootRedisType(Generic[KT, VT], ValueRootType[KT, VT]):
    def __init__(
            self, module: RedisModule, key_fn: Callable[[KT], str]) -> None:
        super().__init__()
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_set_value(self, key: KT, value: VT) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.set(rkey, json_compact(value))

    def do_update_value(self, key: KT, value: VT) -> Optional[VT]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.get(rkey)
                pipe.set(rkey, json_compact(value))
                res = pipe.execute()[0]
                return json_read(res) if res is not None else None

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.setnx(rkey, json_compact(value))
            return bool(res)

    def maybe_get_value(self, key: KT) -> Optional[VT]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.get(rkey)
        return json_read(res) if res is not None else None

    def get_range(
            self,
            prefix: str,
            postfix: Optional[str] = None) -> Iterable[VT]:
        prefix = f"{self._redis.get_prefix()}:{prefix}"
        if postfix is None:
            keys = list(self._redis.keys_str(prefix))
        else:
            keys = [
                key
                for key in self._redis.keys_str(prefix)
                if key.endswith(postfix)
            ]
        with self._redis.get_connection() as conn:
            for res in conn.mget(keys):
                if res is not None:
                    yield json_read(res)


class SetRootRedisType(Generic[KT], SetRootType[KT, str]):
    def __init__(
            self, module: RedisModule, key_fn: Callable[[KT], str]) -> None:
        super().__init__()
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_add_value(self, key: KT, value: str) -> bool:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            with conn.pipeline() as pipe:
                val = value.encode("utf-8")
                pipe.sismember(rkey, val)
                pipe.sadd(rkey, val)
                return bool(pipe.execute()[0])

    def do_remove_value(self, key: KT, value: str) -> bool:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            with conn.pipeline() as pipe:
                val = value.encode("utf-8")
                pipe.sismember(rkey, val)
                pipe.srem(rkey, val)
                return bool(pipe.execute()[0])

    def maybe_get_value(self, key: KT) -> Optional[Set[str]]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            return set(mem.decode("utf-8") for mem in conn.smembers(rkey))


class ValueDependentRedisType(Generic[KT, VT], EffectDependent[KT, VT]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            parents: LT,
            effect: Callable[[EffectDependent[KT, VT], LT, KT], None],
            delay: float) -> None:
        super().__init__(parents, effect, delay)
        self._redis = RedisConnection(module)
        self._key_fn = key_fn

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_set_value(self, key: KT, value: VT) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            conn.set(rkey, json_compact(value))

    def do_update_value(self, key: KT, value: VT) -> Optional[VT]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.get(rkey)
                pipe.set(rkey, json_compact(value))
                res = pipe.execute()[0]
                return json_read(res) if res is not None else None

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.setnx(rkey, json_compact(value))
            return bool(res)

    def retrieve_value(self, key: KT) -> Optional[VT]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.get(rkey)
        return json_read(res) if res is not None else None
