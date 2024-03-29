from typing import Callable, Generic, Iterable, TypeVar

import pandas as pd

from effects.dedicated import LiteralKey, RedisFn, Script
from effects.effects import (
    EffectBase,
    EffectDependent,
    KeyType,
    ListEffectDependent,
    SetRootType,
    ValueRootType,
)
from misc.redis import ConfigKey, RedisConnection, RedisModule
from misc.util import (
    from_timestamp,
    json_compact,
    json_read,
    now_ts,
    to_timestamp,
)
from system.logger.frontend import logger_context


KT = TypeVar('KT', bound=KeyType)
VT = TypeVar('VT')
PT = TypeVar('PT', bound=KeyType)
LT = TypeVar('LT', bound=tuple[EffectBase, ...])


class ValueRootRedisType(Generic[KT, VT], ValueRootType[KT, VT]):
    def __init__(
            self,
            name: str,
            ns_key: ConfigKey,
            module: RedisModule,
            key_fn: Callable[[KT], str]) -> None:
        super().__init__()
        self._name = name
        self._redis = RedisConnection(ns_key, module)
        self._key_fn = key_fn

    def name(self) -> str:
        return self._name

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
                res, _ = pipe.execute()
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

    def get_range_gap_keys(
            self,
            prefix: str,
            gap: int,
            postfix: str | None = None) -> Iterable[str]:
        prefix = f"{self._redis.get_prefix()}:{prefix}"
        fromix = len(prefix)
        toix = None if not postfix else -len(postfix)
        return (
            key[fromix:toix]
            for key in self._redis.keys_gap_str(prefix, gap, postfix)
        )

    def get_keys(
            self, parser: tuple[str, Callable[[str], KT]]) -> Iterable[KT]:
        prefix, parse_key = parser
        yield from (
            parse_key(key)
            for key in sorted(self.get_range_keys(prefix=prefix))
        )

    def key_count(self, parser: tuple[str, Callable[[str], KT]]) -> int:
        prefix, _ = parser
        return self._redis.keys_count(f"{self._redis.get_prefix()}:{prefix}")


class SetRootRedisType(Generic[KT], SetRootType[KT, str]):
    def __init__(
            self,
            name: str,
            ns_key: ConfigKey,
            module: RedisModule,
            key_fn: Callable[[KT], str]) -> None:
        super().__init__()
        self._name = name
        self._redis = RedisConnection(ns_key, module)
        self._key_fn = key_fn

    def name(self) -> str:
        return self._name

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

    def get_size(self, key: KT) -> int:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            return int(conn.scard(rkey))

    def get_keys(
            self, parser: tuple[str, Callable[[str], KT]]) -> Iterable[KT]:
        prefix, parse_key = parser
        prefix = f"{self._redis.get_prefix()}:{prefix}"
        fromix = len(prefix)
        return (
            parse_key(key[fromix:])
            for key in sorted(self._redis.keys_str(prefix, None))
        )

    def key_count(self, parser: tuple[str, Callable[[str], KT]]) -> int:
        prefix, _ = parser
        return self._redis.keys_count(f"{self._redis.get_prefix()}:{prefix}")


class ValueDependentRedisType(Generic[KT, VT], EffectDependent[KT, VT]):
    def __init__(
            self,
            name: str,
            ns_key: ConfigKey,
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
            effect: Callable[[KT, pd.Timestamp | None], None]) -> None:
        super().__init__(parents=parents, effect=effect, convert=convert)
        assert marker_prefix
        assert marker_prefix != value_prefix
        self._name = name
        self._redis = RedisConnection(ns_key, module)
        self._key_fn = key_fn
        self._key_json = key_json
        self._json_key = json_key
        self._value_prefix = value_prefix
        self._marker_prefix = marker_prefix
        self._marker_queue = marker_queue

    def name(self) -> str:
        return self._name

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
        with logger_context({
                    "module": "effects.redis",
                    "type": "value",
                    "name": self._name,
                }) as logger:
            logger.log_count("request_compute")

    def on_set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(mkey, to_timestamp(now))
        with logger_context({
                    "module": "effects.redis",
                    "type": "value",
                    "name": self._name,
                }) as logger:
            logger.log_count("set_outdated")

    def clear_outdated(self, key: KT, now: pd.Timestamp | None) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            when = conn.get(mkey)
            if when is None:
                return
            if now is None or now >= from_timestamp(float(when)):
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
        return (
            json_read(value) if value is not None else None,
            from_timestamp(float(marker)) if marker is not None else None,
        )

    def do_set_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> None:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(rkey, json_compact(value))
        with logger_context({
                    "module": "effects.redis",
                    "type": "value",
                    "name": self._name,
                }) as logger:
            logger.log_count("set_value")

    def do_update_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> VT | None:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.get(rkey)
                pipe.set(rkey, json_compact(value))
                res = pipe.execute()[0]
        with logger_context({
                    "module": "effects.redis",
                    "type": "value",
                    "name": self._name,
                }) as logger:
            logger.log_count("update_value")
        return json_read(res) if res is not None else None

    def do_set_new_value(
            self, key: KT, value: VT, now: pd.Timestamp | None) -> bool:
        rkey = self.get_value_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            res = conn.setnx(rkey, json_compact(value))
        with logger_context({
                    "module": "effects.redis",
                    "type": "value",
                    "name": self._name,
                }) as logger:
            logger.log_count("set_new_value")
        return bool(res)


class ListDependentRedisType(Generic[KT], ListEffectDependent[KT, str]):
    def __init__(
            self,
            name: str,
            ns_key: ConfigKey,
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
            effect: Callable[[KT, pd.Timestamp | None], None],
            empty: bytes | None) -> None:
        super().__init__(parents=parents, effect=effect, convert=convert)
        self._name = name
        self._redis = RedisConnection(ns_key, module)
        self._key_fn = key_fn
        self._key_json = key_json
        self._json_key = json_key
        self._value_prefix = value_prefix
        self._marker_prefix = marker_prefix
        self._marker_queue = marker_queue
        self._update_new_val: Script | None = None
        self._empty = empty
        self._empty_str = "" if empty is None else empty.decode("utf-8")

    def name(self) -> str:
        return self._name

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
        with logger_context({
                    "module": "effects.redis",
                    "type": "list",
                    "name": self._name,
                }) as logger:
            logger.log_count("request_compute")

    def on_set_outdated(self, key: KT, now: pd.Timestamp) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(mkey, to_timestamp(now))
        with logger_context({
                    "module": "effects.redis",
                    "type": "list",
                    "name": self._name,
                }) as logger:
            logger.log_count("set_outdated")

    def clear_outdated(self, key: KT, now: pd.Timestamp | None) -> None:
        # FIXME optimize
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            when = conn.get(mkey)
            if when is None:
                return
            if now is None or now >= from_timestamp(float(when)):
                conn.delete(mkey)

    def _convert_list(self, values: list[bytes]) -> list[str]:
        empty = self._empty
        if empty is None:
            return [val.decode("utf-8") for val in values]
        return [val.decode("utf-8") for val in values if val != empty]

    def _prepare_list(self, value: list[str]) -> list[bytes]:
        res = [val.encode("utf-8") for val in value]
        empty = self._empty
        if empty is None:
            return res
        if empty in res:
            raise ValueError(f"cannot include empty marker in value: {value}")
        if not value and empty is not None:
            return [empty]
        return res

    def _prepare_list_str(self, value: list[str]) -> list[str]:
        empty = self._empty
        if (empty is not None
                and empty in [val.encode("utf-8") for val in value]):
            raise ValueError(f"cannot include empty marker in value: {value}")
        if not value and empty is not None:
            return [self._empty_str]
        return value

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
        return (
            self._convert_list(value) if int(has) else None,
            from_timestamp(float(marker)) if marker is not None else None,
        )

    def do_set_value(
            self, key: KT, value: list[str], now: pd.Timestamp | None) -> None:
        rkey = self.get_value_redis_key(key)
        pval = self._prepare_list(value)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.delete(rkey)
                if pval:
                    pipe.rpush(rkey, *pval)
                pipe.execute()
        with logger_context({
                    "module": "effects.redis",
                    "type": "list",
                    "name": self._name,
                }) as logger:
            logger.log_count("set_value")

    def do_update_value(
            self,
            key: KT,
            value: list[str],
            now: pd.Timestamp | None) -> list[str] | None:
        rkey = self.get_value_redis_key(key)
        pval = self._prepare_list(value)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.exists(rkey)
                pipe.lrange(rkey, 0, -1)
                pipe.delete(rkey)
                if pval:
                    pipe.rpush(rkey, *pval)
                exec_res = pipe.execute()
        has = exec_res[0]
        res = exec_res[1]
        with logger_context({
                    "module": "effects.redis",
                    "type": "list",
                    "name": self._name,
                }) as logger:
            logger.log_count("update_value")
        if not int(has):
            return None
        return self._convert_list(res)

    def do_set_new_value(
            self, key: KT, value: list[str], now: pd.Timestamp | None) -> bool:
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
            args={"value": self._prepare_list_str(value)},
            keys={"rkey": rkey},
            now=now_ts() if now is None else now,
            conn=self._redis,
            depth=1)
        with logger_context({
                    "module": "effects.redis",
                    "type": "list",
                    "name": self._name,
                }) as logger:
            logger.log_count("set_new_value")
        return int(res) != 0

    def do_get_value_range(
            self,
            key: KT,
            from_ix: int | None,
            to_ix: int | None) -> tuple[list[str] | None, pd.Timestamp | None]:
        rkey = self.get_value_redis_key(key)
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            if to_ix == 0:
                res = []
                with conn.pipeline() as pipe:
                    pipe.exists(rkey)
                    pipe.get(mkey)
                    has, marker = pipe.execute()
            else:
                if from_ix is None:
                    from_ix = 0
                if to_ix is None:
                    to_ix = -1
                else:
                    to_ix -= 1
                with conn.pipeline() as pipe:
                    pipe.exists(rkey)
                    pipe.lrange(rkey, from_ix, to_ix)
                    pipe.get(mkey)
                    has, res, marker = pipe.execute()
        return (
            self._convert_list(res) if int(has) else None,
            from_timestamp(float(marker)) if marker is not None else None,
        )

    def do_get_size(self, key: KT) -> tuple[int | None, pd.Timestamp | None]:
        # FIXME optimize
        rkey = self.get_value_redis_key(key)
        mkey = self.get_marker_redis_key(key)
        with self._redis.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.exists(rkey)
                pipe.llen(rkey)
                pipe.lrange(rkey, 0, 0)
                pipe.get(mkey)
                has, res, content, marker = pipe.execute()
        if not self._convert_list(content):
            res = 0
        return (
            int(res) if int(has) else None,
            from_timestamp(float(marker)) if marker is not None else None,
        )
