import contextlib
import os
import threading
import time
import uuid
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    Literal,
    overload,
    Protocol,
)

from redis import StrictRedis
from redis.client import Script
from redis.exceptions import ResponseError
from redis_lock import Lock

from misc.env import envload_int, envload_str
from misc.io import open_read
from misc.util import get_relative_function_info, json_compact, json_read


REDIS_SALT_LOCK = threading.RLock()
REDIS_SALT: dict[str, str] = {}

REDIS_SLOW = 0.5
REDIS_SLOW_CONTEXT = 3
REDIS_UNIQUE: set[tuple[str, int, str]] = set()
NL = "\n"


def is_test() -> bool:
    test_id = os.getenv("PYTEST_CURRENT_TEST")
    return test_id is not None


def get_salt() -> str | None:
    test_id = os.getenv("PYTEST_CURRENT_TEST")
    if test_id is None:
        return None
    res = REDIS_SALT.get(test_id)
    if res is None:
        with REDIS_SALT_LOCK:
            res = REDIS_SALT.get(test_id)
            if res is None:
                res = f"salt:{uuid.uuid4().hex}"
                REDIS_SALT[test_id] = res
    return res


class RedisFunctionBytes(Protocol):  # pylint: disable=too-few-public-methods
    def __call__(
            self,
            *,
            keys: list[str],
            args: list[Any],
            client: StrictRedis | None,
            depth: int) -> bytes:
        ...


class RedisFunctionList(Protocol):  # pylint: disable=too-few-public-methods
    def __call__(
            self,
            *,
            keys: list[str],
            args: list[Any],
            client: StrictRedis | None,
            depth: int) -> list[bytes]:
        ...


RedisModule = Literal[
    "link",
    "locks",
    "token",
    "test",
]
LOCK_MODULE: RedisModule = "locks"


SCRIPT_CACHE: dict[str, str] = {}
SCRIPT_UTILS_CACHE: str = ""
SCRIPT_UTILS_LINES: int = 0
CONCURRENT_MODULE_CONN: int = 17
REDIS_SERVICE_CONN: dict[str, StrictRedis | None] = {}
DO_CACHE = True
LOCK = threading.RLock()
LOCKS: dict[str, tuple[threading.RLock, Lock] | None] = {}
LOCK_ID: str = uuid.uuid4().hex


class RedisWrapper:
    def __init__(self, module: RedisModule) -> None:
        self._module = module
        self._conn = None

    @staticmethod
    def _create_connection() -> StrictRedis:
        config = {
            "retry_on_timeout": True,
            "health_check_interval": 45,
            "client_name": f"api-{uuid.uuid4().hex}",
        }
        return StrictRedis(  # pylint: disable=unexpected-keyword-arg
            host=envload_str("REDIS_HOST", default="localhost"),
            port=envload_int("REDIS_PORT", default=6379),
            db=0,
            password=envload_str("REDIS_PASS", default=""),
            **config)

    @classmethod
    def _get_redis_cached_conn(cls, module: RedisModule) -> StrictRedis:
        if not DO_CACHE:
            return cls._create_connection()

        key = f"{module}-{threading.get_ident() % CONCURRENT_MODULE_CONN}"
        res = REDIS_SERVICE_CONN.get(key)
        if res is None:
            with LOCK:
                if res is None:
                    res = cls._create_connection()
                    REDIS_SERVICE_CONN[key] = res
        return res

    @classmethod
    def _get_lock_pair(cls, name: str) -> tuple[threading.RLock, Lock]:
        res = LOCKS.get(name)
        if res is None:
            with LOCK:
                res = LOCKS.get(name)
                if res is None:
                    res = (
                        threading.RLock(),
                        Lock(
                            cls._get_redis_cached_conn(LOCK_MODULE),
                            name,
                            id=LOCK_ID,
                            expire=1,
                            auto_renewal=True,
                        ),
                    )
                    LOCKS[name] = res
        return res

    @contextlib.contextmanager
    def get_connection(self, *, depth: int) -> Iterator[StrictRedis]:
        conn_start = time.monotonic()
        try:
            if self._conn is None:
                self._conn = self._get_redis_cached_conn(self._module)
            yield self._conn
        except Exception:
            self.reset()
            raise
        finally:
            conn_time = time.monotonic() - conn_start
            if conn_time > REDIS_SLOW:
                fun_info = get_relative_function_info(depth=depth + 1)
                fun_key = fun_info[:3]
                if fun_key not in REDIS_UNIQUE:
                    fun_fname, fun_line, fun_name, fun_locals = fun_info
                    context = []
                    try:
                        with open_read(fun_fname, text=True) as fin:
                            for lineno, line in enumerate(fin):
                                if lineno < fun_line - REDIS_SLOW_CONTEXT:
                                    continue
                                if lineno > fun_line + REDIS_SLOW_CONTEXT:
                                    break
                                if lineno == fun_line:
                                    context.append(f"> {line.rstrip()}")
                                else:
                                    context.append(f"  {line.rstrip()}")
                    except FileNotFoundError:
                        context.append("## not available ##")
                    print(
                        f"slow redis call ({conn_time:.2f}s) "
                        f"at {fun_name} ({fun_fname}:{fun_line})\n"
                        f"{NL.join(context)}\nlocals:\n{fun_locals}")
                    REDIS_UNIQUE.add(fun_key)

    def reset(self) -> None:
        conn = self._conn
        if conn is not None:
            self._invalidate_connection(self._module)
            self._conn = None

    @classmethod
    def reset_lock(cls) -> None:
        with LOCK:
            LOCKS.clear()
            cls._invalidate_connection(LOCK_MODULE)

    @staticmethod
    def _invalidate_connection(module: RedisModule) -> None:
        with LOCK:
            key = f"{module}-{threading.get_ident() % CONCURRENT_MODULE_CONN}"
            REDIS_SERVICE_CONN[key] = None

    @classmethod
    @contextlib.contextmanager
    def create_lock(cls, name: str) -> Iterator[Lock]:
        try:
            local_lock, redis_lock = cls._get_lock_pair(name)
            with local_lock:
                yield redis_lock
        except Exception:
            cls.reset_lock()
            raise

    @staticmethod
    @contextlib.contextmanager
    def no_lock() -> Iterator[None]:
        yield


class RedisConnection:
    def __init__(self, module: RedisModule) -> None:
        self._conn = RedisWrapper(module)
        salt = get_salt()
        salt_str = "" if salt is None else f"{salt}:"
        self._module = f"{salt_str}{module}"

    def get_connection(self, *, depth: int) -> ContextManager[StrictRedis]:
        return self._conn.get_connection(depth=depth + 1)

    def get_dynamic_script(self, code: str) -> RedisFunctionBytes:
        if is_test():
            print(
                "Compiled script:\n-- SCRIPT START\n"
                f"{code.rstrip()}\n-- SCRIPT END")
        compute = Script(None, code.encode("utf-8"))
        context = 3

        def get_error(err_msg: str) -> tuple[str, list[str]] | None:
            ustr = "user_script:"
            ix = err_msg.find(ustr)
            if ix < 0:
                return None
            eix = err_msg.find(":", ix + len(ustr))
            if eix < 0:
                return None
            num = int(err_msg[ix + len(ustr):eix])
            rel_line = num

            new_msg = f"{err_msg[:ix]}:{rel_line}{err_msg[eix:]}"
            ctx = code.splitlines()
            return new_msg, ctx[max(num - context - 1, 0):num + context]

        @contextlib.contextmanager
        def get_client(
                *,
                client: StrictRedis | None,
                depth: int) -> Iterator[StrictRedis]:
            try:
                if client is None:
                    with self.get_connection(depth=depth + 1) as res:
                        yield res
                else:
                    yield client
            except ResponseError as e:
                handle_err(e)
                raise e

        def handle_err(exc: ResponseError) -> None:
            if exc.args:
                msg = exc.args[0]
                res = get_error(msg)
                if res is not None:
                    ctx = "\n".join((
                        f"{'>' if ix == context else ' '} {line}"
                        for (ix, line) in enumerate(res[1])))
                    exc.args = (
                        f"{res[0].rstrip()}\nCode:\n{code}\nContext:\n{ctx}",
                    )

        def execute_bytes_result(
                *,
                keys: list[str],
                args: list[bytes | str | int],
                client: StrictRedis | None,
                depth: int) -> bytes:
            with get_client(client=client, depth=depth + 1) as inner:
                return compute(keys=keys, args=args, client=inner)

        return execute_bytes_result

    @overload
    def get_script(
            self,
            filename: str,
            return_list: Literal[False]) -> RedisFunctionBytes:
        ...

    @overload
    def get_script(
            self,
            filename: str,
            return_list: Literal[True]) -> RedisFunctionList:
        ...

    def get_script(
            self,
            filename: str,
            return_list: bool) -> RedisFunctionBytes | RedisFunctionList:
        global SCRIPT_UTILS_CACHE, SCRIPT_UTILS_LINES

        script_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "scripts"))
        utils = "utils.lua"
        fname = f"{filename}.lua"
        if not SCRIPT_UTILS_CACHE:
            with LOCK:
                if not SCRIPT_UTILS_CACHE:
                    script_utils_path = os.path.join(script_root, utils)
                    with open_read(script_utils_path, text=True) as fin:
                        SCRIPT_UTILS_CACHE = fin.read().rstrip()
                    SCRIPT_UTILS_LINES = len(SCRIPT_UTILS_CACHE.splitlines())

        if filename not in SCRIPT_CACHE:
            with LOCK:
                if filename not in SCRIPT_CACHE:
                    script_path = os.path.join(script_root, fname)
                    with open_read(script_path, text=True) as fin:
                        script_txt = f"{SCRIPT_UTILS_CACHE}\n{fin.read()}"
                        SCRIPT_CACHE[filename] = script_txt
        script_txt = SCRIPT_CACHE[filename]
        compute = Script(None, script_txt.encode("utf-8"))
        context = 2

        def get_error(err_msg: str) -> tuple[str, list[str]] | None:
            ustr = "user_script:"
            ix = err_msg.find(ustr)
            if ix < 0:
                return None
            eix = err_msg.find(":", ix + len(ustr))
            if eix < 0:
                return None
            num = int(err_msg[ix + len(ustr):eix])
            if num < SCRIPT_UTILS_LINES:
                name = utils
                rel_line = num
            else:
                name = fname
                rel_line = num - SCRIPT_UTILS_LINES
            new_msg = f"{err_msg[:ix]}{name}:{rel_line}{err_msg[eix:]}"
            ctx = script_txt.splitlines()
            return new_msg, ctx[max(num - context - 1, 0):num + context]

        @contextlib.contextmanager
        def get_client(
                *,
                client: StrictRedis | None,
                depth: int) -> Iterator[StrictRedis]:
            try:
                if client is None:
                    with self.get_connection(depth=depth + 1) as res:
                        yield res
                else:
                    yield client
            except ResponseError as e:
                handle_err(e)
                raise e

        def handle_err(exc: ResponseError) -> None:
            if exc.args:
                msg = exc.args[0]
                res = get_error(msg)
                if res is not None:
                    ctx = "\n".join((
                        f"{'>' if ix == context else ' '} {line}"
                        for (ix, line) in enumerate(res[1])))
                    exc.args = (f"{res[0].rstrip()}\nContext:\n{ctx}",)

        def execute_list_result(
                *,
                keys: list[str],
                args: list[bytes | str | int],
                client: StrictRedis | None,
                depth: int) -> list[bytes]:
            with get_client(client=client, depth=depth + 1) as inner:
                res = compute(keys=keys, args=args, client=inner)
                assert isinstance(res, list)
                return res

        def execute_bytes_result(
                *,
                keys: list[str],
                args: list[bytes | str | int],
                client: StrictRedis | None,
                depth: int) -> bytes:
            with get_client(client=client, depth=depth + 1) as inner:
                return compute(keys=keys, args=args, client=inner)

        if return_list:
            return execute_list_result
        return execute_bytes_result

    def get_prefix(self) -> str:
        return f"api:{self._module}"

    def get_pubsub_key(self, key: str) -> str:
        return f"apips:{self._module}:{key}"

    def wait_for(
            self,
            key: str,
            predicate: Callable[[], bool],
            granularity: float = 30.0) -> None:
        if predicate():
            return
        with self.get_connection(depth=1) as conn:
            with conn.pubsub() as psub:
                psub.subscribe(self.get_pubsub_key(key))
                try:
                    while not predicate():
                        psub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=granularity)
                        while psub.get_message() is not None:  # flushing queue
                            pass
                finally:
                    psub.unsubscribe()

    def notify_all(self, key: str) -> None:
        with self.get_connection(depth=1) as conn:
            conn.publish(self.get_pubsub_key(key), "notify")

    def ping(self) -> None:
        with self.get_connection(depth=1) as conn:
            conn.ping()

    def flush_all(self) -> None:
        with self.get_connection(depth=1) as conn:
            conn.flushall()

    def keys_count(self, prefix: str) -> int:
        full_prefix = f"{prefix}*"
        vals: set[bytes] = set()
        cursor = 0
        count = 10
        with self.get_connection(depth=1) as conn:
            while True:
                cursor, res = conn.scan(cursor, full_prefix, count)
                vals.update(res)
                if cursor == 0:
                    break
                if count < 4000:
                    count = min(4000, count * 2)
        return len(vals)

    def keys_count_approx(self, prefix: str) -> int:
        return self.keys_count(prefix)

    def keys_str(
            self, prefix: str, postfix: str | None = None) -> list[str]:
        full_prefix = f"{prefix}*{'' if postfix is None else postfix}"
        vals: set[bytes] = set()
        cursor = 0
        count = 10
        with self.get_connection(depth=1) as conn:
            while True:
                cursor, res = conn.scan(cursor, full_prefix, count)
                vals.update(res)
                if cursor == 0:
                    break
                if count < 4000:
                    count = min(4000, count * 2)
        return [val.decode("utf-8") for val in vals]

    def prefix_exists(
            self, prefix: str, postfix: str | None = None) -> bool:
        full_prefix = f"{prefix}*{'' if postfix is None else postfix}"
        cursor = 0
        count = 10
        with self.get_connection(depth=1) as conn:
            while True:
                cursor, res = conn.scan(cursor, full_prefix, count)
                if res:
                    return True
                if cursor == 0:
                    return False
                if count < 4000:
                    count = min(4000, count * 2)

    @contextlib.contextmanager
    def get_lock(self, name: str) -> Iterator[None]:
        with self._conn.create_lock(f"{self.get_prefix()}:{name}"):
            yield


class ObjectRedis(RedisConnection):
    def __init__(self, module: RedisModule) -> None:
        super().__init__(module)
        self._objs = f"{self.get_prefix()}:objects"

        # LUA scripts
        self._flush_keys = self.get_script("obj_flush_keys", return_list=False)
        self._obj_put = self.get_script("obj_put", return_list=False)

    def compute_name(self, name: str, key: str = "*") -> str:
        return f"{self._objs}:{name}:{key}"

    def obj_flush_keys(self, patterns: list[str], skip: list[str]) -> None:
        self._flush_keys(
            keys=[],
            args=[json_compact(patterns), *skip],
            client=None,
            depth=1)

    def obj_put(
            self,
            name: str,
            key: str,
            value: Any,
            preserve_expire: bool = False) -> None:
        """
        Sets path (f(name, key)) with value. Can preserve ttl of this key.
        """
        path = self.compute_name(name, key)
        self._obj_put(
            keys=[path],
            args=[int(preserve_expire), json_compact(value)],
            client=None,
            depth=1)

    def obj_put_nx(
            self,
            name: str,
            key: str,
            value: Any) -> bool:
        """
        Sets path (f(name, key)) with value if path does not exist yet.
        Returns True if the value was set.
        """
        path = self.compute_name(name, key)
        with self.get_connection(depth=1) as conn:
            res = int(conn.setnx(path, json_compact(value)))
        return res != 0

    def obj_put_expire(
            self,
            name: str,
            key: str,
            value: Any,
            expire: float | None) -> None:
        path = self.compute_name(name, key)
        with self.get_connection(depth=1) as conn:
            conn.set(
                path,
                json_compact(value),
                px=None if expire is None else int(expire * 1000))

    def obj_expire(self, name: str, key: str, expire: float) -> None:
        with self.get_connection(depth=1) as conn:
            conn.pexpire(self.compute_name(name, key), int(expire * 1000))

    def obj_get(self, name: str, key: str, default: Any = None) -> Any:
        with self.get_connection(depth=1) as conn:
            res = conn.get(self.compute_name(name, key))
        return json_read(res) if res is not None else default

    def obj_get_expire(
            self,
            name: str,
            key: str,
            expire: float | None,
            default: Any = None) -> Any:
        path = self.compute_name(name, key)
        with self.get_connection(depth=1) as conn:
            # expire before reading to ensure correct treatment of 0
            if expire is not None:
                conn.pexpire(path, int(expire * 1000))
            res = conn.get(path)
        return json_read(res) if res is not None else default

    def obj_ttl(self, name: str, key: str) -> float | None:
        # pylint: disable=unbalanced-tuple-unpacking

        path = self.compute_name(name, key)
        with self.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.exists(path)
                pipe.pttl(path)
                exists, pexpire = pipe.execute()
        if exists:
            return max(pexpire / 1000.0, 0.0)
        return None

    def obj_has(self, name: str, key: str) -> bool:
        with self.get_connection(depth=1) as conn:
            return conn.get(self.compute_name(name, key)) is not None

    def obj_keys(self, name: str) -> list[str]:
        path = self.compute_name(name, key="")
        return [
            key[len(path):]
            for key in self.keys_str(path)
        ]

    def obj_dict(self, name: str) -> dict[str, Any]:
        path = self.compute_name(name, key="")
        keys = self.keys_str(path)
        with self.get_connection(depth=1) as conn:
            return {
                key[len(path):]: json_read(res)
                for (key, res) in zip(keys, conn.mget(keys))
                if res is not None
            }

    def obj_keys_count(self, name: str) -> int:
        path = self.compute_name(name, key="")
        return self.keys_count(path)

    def obj_partial_keys(self, partial: str) -> list[str]:
        path = f"{self._objs}:{partial}"
        return [
            key[len(path):]
            for key in self.keys_str(path)
        ]

    def obj_remove(self, name: str, key: str = "*") -> None:
        with self.get_connection(depth=1) as conn:
            conn.delete(self.compute_name(name, key))

    def obj_remove_all(self, name: str, keys: list[str]) -> None:
        with self.get_connection(depth=1) as conn:
            for key in keys:
                conn.delete(self.compute_name(name, key))

    def obj_pop_raw(self, name: str, key: str) -> bytes | None:
        path = self.compute_name(name, key)
        with self.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.get(path)
                pipe.delete(path)
                return pipe.execute()[0]

    def obj_pop(self, name: str, key: str, default: Any = None) -> Any:
        path = self.compute_name(name, key)
        with self.get_connection(depth=1) as conn:
            with conn.pipeline() as pipe:
                pipe.get(path)
                pipe.delete(path)
                res = pipe.execute()[0]
        return json_read(res) if res is not None else default
