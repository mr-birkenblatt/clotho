from typing import Callable, Generic, Iterable, TypeVar

import pandas as pd

from effects.dedicated import LiteralKey, RedisFn, Script
from effects.effects import (
    EffectBase,
    EffectDependent,
    KeyType,
    SetRootType,
    ValueRootType,
)
from misc.redis import RedisConnection, RedisModule
from misc.util import from_timestamp, json_compact, json_read, to_timestamp


KT = TypeVar('KT', bound=KeyType)
VT = TypeVar('VT')
PT = TypeVar('PT', bound=KeyType)
LT = TypeVar('LT', bound=tuple[EffectBase, ...])


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
        with self._redis.get_connection(depth=1) as conn:
            conn.set(rkey, json_compact(value))

    def do_update_value(self, key: KT, value: VT) -> VT | None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.get(rkey)
                pipe.set(rkey, json_compact(value))
                res = pipe.execute()[0]
                return json_read(res) if res is not None else None

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            res = conn.setnx(rkey, json_compact(value))
            return bool(res)

    def maybe_get_value(self, key: KT) -> VT | None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            res = conn.get(rkey)
        return json_read(res) if res is not None else None

    def get_range(
            self,
            prefix: str,
            postfix: str | None = None) -> Iterable[VT]:
        prefix = f"{self._redis.get_prefix()}:{prefix}"
        keys = list(self._redis.keys_str(prefix, postfix))
        with self._redis.get_connection(depth=1) as conn:
            return (
                json_read(res)
                for res in conn.mget(keys)
                if res is not None
            )

    def get_range_keys(
            self,
            prefix: str,
            postfix: str | None = None) -> Iterable[str]:
        prefix = f"{self._redis.get_prefix()}:{prefix}"
        fromix = len(prefix)
        toix = None if not postfix else -len(postfix)
        return (
            key[fromix:toix]
            for key in self._redis.keys_str(prefix, postfix)
        )


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
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                val = value.encode("utf-8")
                pipe.sismember(rkey, val)
                pipe.sadd(rkey, val)
                return bool(pipe.execute()[0])

    def do_remove_value(self, key: KT, value: str) -> bool:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                val = value.encode("utf-8")
                pipe.sismember(rkey, val)
                pipe.srem(rkey, val)
                return bool(pipe.execute()[0])

    def maybe_get_value(self, key: KT) -> set[str] | None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            return set(mem.decode("utf-8") for mem in conn.smembers(rkey))

    def has_value(self, key: KT, value: str) -> bool:
        rkey = self.get_redis_key(key)
        val = value.encode("utf-8")
        with self._redis.get_connection(depth=1) as conn:
            return bool(conn.sismember(rkey, val))


class ValueDependentRedisType(Generic[KT, VT], EffectDependent[KT, VT]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            key_json: Callable[[KT], bytes],
            json_key: Callable[[bytes], KT],
            value_prefix: str,
            marker_prefix: str,
            marker_queue: str,
            *,
            parents: tuple[EffectBase[PT], ...],
            convert: Callable[[PT], KT],
            effect: Callable[[KT], None]) -> None:
        super().__init__(parents=parents, effect=effect, convert=convert)
        assert marker_prefix
        assert marker_prefix != value_prefix
        self._redis = RedisConnection(module)
        self._key_fn = key_fn
        self._key_json = key_json
        self._json_key = json_key
        self._value_prefix = value_prefix
        self._marker_prefix = marker_prefix
        self._marker_queue = marker_queue

    def get_value_redis_key(self, key: KT) -> str:
        value_prefix = self._value_prefix
        if value_prefix:
            value_prefix = f":{value_prefix}"
        return f"{self._redis.get_prefix()}{value_prefix}:{self._key_fn(key)}"

    def get_marker_redis_key(self, key: KT) -> str:
        return (
            f"{self._redis.get_prefix()}:"
            f"{self._marker_prefix}:"
            f"{self._key_fn(key)}"
        )

    def get_marker_queue_redis_key(self) -> str:
        return f"{self._redis.get_prefix()}:{self._marker_queue}"

    def pending_outdated(self) -> tuple[KT, pd.Timestamp] | None:
        # FIXME optimize
        key_queue = self.get_marker_queue_redis_key()
        with self._redis.get_connection(depth=1) as conn:
            while True:
                cur = conn.lpop(key_queue)
                if cur is None:
                    return None
                res_key = self._json_key(cur)
                mkey = self.get_marker_redis_key(res_key)
                when_marker = conn.get(mkey)
                if when_marker is None:
                    continue
                return res_key, from_timestamp(float(when_marker))

    def on_request_compute(self, key: KT) -> None:
        # FIXME optimize
        key_queue = self.get_marker_queue_redis_key()
        with self._redis.get_connection(depth=1) as conn:
            conn.rpush(key_queue, self._key_json(key))

    def set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(mkey, to_timestamp(now))

    def clear_outdated(self, key: KT, now: pd.Timestamp) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            when = conn.get(mkey)
            if when is None:
                return
            if now > from_timestamp(float(when)):
                conn.delete(mkey)

    def retrieve_value(self, key: KT) -> tuple[VT | None, pd.Timestamp | None]:
        # FIXME optimize
        vkey = self.get_value_redis_key(key)
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.get(vkey)
                pipe.get(mkey)
                value, marker = pipe.execute()
        self.maybe_compute(key, marker)
        return (
            json_read(value) if value is not None else None,
            from_timestamp(float(marker)) if marker is not None else None,
        )

    def do_set_value(self, key: KT, value: VT) -> None:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(rkey, json_compact(value))

    def do_update_value(self, key: KT, value: VT) -> VT | None:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.get(rkey)
                pipe.set(rkey, json_compact(value))
                res = pipe.execute()[0]
                return json_read(res) if res is not None else None

    def do_set_new_value(self, key: KT, value: VT) -> bool:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            res = conn.setnx(rkey, json_compact(value))
            return bool(res)


class ListDependentRedisType(Generic[KT], EffectDependent[KT, list[str]]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            key_json: Callable[[KT], bytes],
            json_key: Callable[[bytes], KT],
            value_prefix: str,
            marker_prefix: str,
            marker_queue: str,
            *,
            parents: tuple[EffectBase[PT], ...],
            convert: Callable[[PT], KT],
            effect: Callable[[KT], None]) -> None:
        super().__init__(parents=parents, effect=effect, convert=convert)
        self._redis = RedisConnection(module)
        self._key_fn = key_fn
        self._key_json = key_json
        self._json_key = json_key
        self._value_prefix = value_prefix
        self._marker_prefix = marker_prefix
        self._marker_queue = marker_queue
        self._update_new_val: Script | None = None

    def get_value_redis_key(self, key: KT) -> str:
        value_prefix = self._value_prefix
        if value_prefix:
            value_prefix = f":{value_prefix}"
        return f"{self._redis.get_prefix()}{value_prefix}:{self._key_fn(key)}"

    def get_marker_redis_key(self, key: KT) -> str:
        return (
            f"{self._redis.get_prefix()}:"
            f"{self._marker_prefix}:"
            f"{self._key_fn(key)}"
        )

    def get_marker_queue_redis_key(self) -> str:
        return f"{self._redis.get_prefix()}:{self._marker_queue}"

    def pending_outdated(self) -> tuple[KT, pd.Timestamp] | None:
        # FIXME optimize
        key_queue = self.get_marker_queue_redis_key()
        with self._redis.get_connection(depth=1) as conn:
            while True:
                cur = conn.lpop(key_queue)
                if cur is None:
                    return None
                res_key = self._json_key(cur)
                mkey = self.get_marker_redis_key(res_key)
                when_marker = conn.get(mkey)
                if when_marker is None:
                    continue
                return res_key, from_timestamp(float(when_marker))

    def on_request_compute(self, key: KT) -> None:
        # FIXME optimize
        key_queue = self.get_marker_queue_redis_key()
        with self._redis.get_connection(depth=1) as conn:
            conn.rpush(key_queue, self._key_json(key))

    def set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(mkey, to_timestamp(now))

    def clear_outdated(self, key: KT, now: pd.Timestamp) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            when = conn.get(mkey)
            if when is None:
                return
            if now > from_timestamp(float(when)):
                conn.delete(mkey)

    def retrieve_value(
            self, key: KT) -> tuple[list[str] | None, pd.Timestamp | None]:
        # FIXME optimize
        vkey = self.get_value_redis_key(key)
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.exists(vkey)
                pipe.lrange(vkey, 0, -1)
                pipe.get(mkey)
                has, value, marker = pipe.execute()
        self.maybe_compute(key, marker)
        return (
            [val.decode("utf-8") for val in value] if int(has) else None,
            from_timestamp(float(marker)) if marker is not None else None,
        )

    def do_set_value(self, key: KT, value: list[str]) -> None:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.delete(rkey)
                if value:
                    pipe.rpush(rkey, *[val.encode("utf-8") for val in value])
                pipe.execute()

    def do_update_value(self, key: KT, value: list[str]) -> list[str] | None:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.exists(rkey)
                pipe.lrange(rkey, 0, -1)
                pipe.delete(rkey)
                if value:
                    pipe.rpush(rkey, *[val.encode("utf-8") for val in value])
                has, res, _, _ = pipe.execute()
                if not int(has):
                    return None
                return [val.decode("utf-8") for val in res]

    def do_set_new_value(self, key: KT, value: list[str]) -> bool:
        if not value:
            return False
        if self._update_new_val is None:
            script = Script()
            new_value = script.add_arg("value")
            key_var = script.add_key("rkey", LiteralKey())
            res_var = script.add_local(0)

            success, _ = script.if_(RedisFn("LLEN", key_var).eq(0))
            loop, _, loop_value = success.for_(new_value)
            loop.add(RedisFn("RPUSH", key_var, loop_value))
            success.add(res_var.assign(1))

            script.set_return_value(res_var)
            self._update_new_val = script
        rkey = self.get_value_redis_key(key)
        res = self._update_new_val.execute(
            args={"value": value},
            keys={"rkey": rkey},
            conn=self._redis,
            depth=1)
        return int(res) != 0

    def get_value_range(
            self,
            key: KT,
            from_ix: int | None,
            to_ix: int | None) -> list[str]:
        mkey = self.get_marker_redis_key(key)
        if to_ix == 0:
            res = []
            with self._redis.get_connection(depth=1) as conn:
                marker = conn.get(mkey)
        else:
            if from_ix is None:
                from_ix = 0
            if to_ix is None:
                to_ix = -1
            else:
                to_ix -= 1
            rkey = self.get_value_redis_key(key)
            with self._redis.get_connection(depth=1) as conn:
                with conn.pipeline() as pipe:
                    pipe.lrange(rkey, from_ix, to_ix)
                    pipe.get(mkey)
                    res, marker = pipe.execute()
        self.maybe_compute(key, marker)
        return [val.decode("utf-8") for val in res]

    def __getitem__(self, arg: tuple[KT, slice]) -> list[str]:
        key, slicer = arg
        if slicer.step is None or slicer.step > 0:
            res = self.get_value_range(key, slicer.start, slicer.stop)
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
            res = self.get_value_range(key, flip_start, flip_stop)
        if slicer.step is not None and slicer.step != 1:
            return res[::slicer.step]
        return res
