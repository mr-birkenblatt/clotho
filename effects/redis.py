from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from effects.dedicated import (
    Arg,
    Branch,
    ForLoop,
    Literal,
    LiteralKey,
    LocalVariable,
    NotOp,
    RedisFn,
    Script,
)
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
PT = TypeVar('PT')
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

    def get_range_keys(
            self,
            prefix: str,
            postfix: Optional[str] = None) -> Iterable[str]:
        prefix = f"{self._redis.get_prefix()}:{prefix}"
        fromix = len(prefix)
        toix = None if not postfix else -len(postfix)
        if postfix is None:
            keys = list(self._redis.keys_str(prefix))
        else:
            keys = [
                key
                for key in self._redis.keys_str(prefix)
                if key.endswith(postfix)
            ]
        for key in keys:
            yield key[fromix:toix]


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


class ValueDependentRedisType(
        Generic[KT, VT, PT], EffectDependent[KT, VT, PT]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            parents: LT,
            effect: Callable[[EffectDependent[KT, VT, PT], LT, PT, KT], None],
            conversion: Callable[[PT], KT],
            delay: float) -> None:
        super().__init__(parents, effect, conversion, delay)
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


class ListDependentRedisType(
        Generic[KT, PT], EffectDependent[KT, List[str], PT]):
    def __init__(
            self,
            module: RedisModule,
            key_fn: Callable[[KT], str],
            parents: LT,
            effect: Callable[
                [EffectDependent[KT, List[str], PT], LT, PT, KT], None],
            conversion: Callable[[PT], KT],
            delay: float) -> None:
        super().__init__(parents, effect, conversion, delay)
        self._redis = RedisConnection(module)
        self._key_fn = key_fn
        self._update_new_val: Optional[Script] = None

    def get_redis_key(self, key: KT) -> str:
        return f"{self._redis.get_prefix()}:{self._key_fn(key)}"

    def do_set_value(self, key: KT, value: List[str]) -> None:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.delete(rkey)
                if value:
                    pipe.rpush(rkey, *[val.encode("utf-8") for val in value])
                pipe.execute()

    def do_update_value(self, key: KT, value: List[str]) -> Optional[List[VT]]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
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

    def do_set_new_value(self, key: KT, value: List[str]) -> bool:
        if not value:
            return False
        if self._update_new_val is None:
            script = Script()
            new_value = script.add_arg(Arg())
            key_var = script.add_key(LiteralKey())
            res_var = script.add_local(LocalVariable(Literal(0)))

            branch = Branch(NotOp(RedisFn("EXISTS", key_var)))
            loop = ForLoop(script, new_value)
            loop.get_loop().add_stmt(RedisFn(
                "RPUSH", key_var, loop.get_value()).as_stmt())
            branch.get_success().add_stmt(
                loop).add_stmt(res_var.assign(Literal(1)))

            script.set_return_value(res_var)
            self._update_new_val = script
        rkey = self.get_redis_key(key)
        return int(self._update_new_val.execute(
            args=[value], keys=[rkey], conn=self._redis)) != 0

    def retrieve_value(self, key: KT) -> Optional[List[str]]:
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.exists(rkey)
                pipe.lrange(rkey, 0, -1)
                has, res = pipe.execute()
        if not int(has):
            return None
        return [val.decode("utf-8") for val in res]

    def get_value_range(
            self,
            key: KT,
            from_ix: Optional[int],
            to_ix: Optional[int]) -> List[str]:
        if to_ix == 0:
            return []
        if from_ix is None:
            from_ix = 0
        if to_ix is None:
            to_ix = -1
        else:
            to_ix -= 1
        rkey = self.get_redis_key(key)
        with self._redis.get_connection() as conn:
            res = conn.lrange(rkey, from_ix, to_ix)
        return [val.decode("utf-8") for val in res]

    def __getitem__(self, arg: Tuple[KT, slice]) -> List[str]:
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
