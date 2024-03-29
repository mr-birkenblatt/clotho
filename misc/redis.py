import contextlib
import os
import threading
import time
import uuid
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    overload,
    Protocol,
    TypedDict,
)

from redis import StrictRedis
from redis.client import Script
from redis.exceptions import ResponseError
from redis_lock import Lock

from misc.env import envload_int, envload_path, envload_str
from misc.io import open_read
from misc.util import (
    get_relative_function_info,
    get_test_salt,
    is_test,
    json_compact,
    json_pretty,
    json_read,
    NL,
)


RedisConfig = TypedDict('RedisConfig', {
    "host": str,
    "port": int,
    "passwd": str,
    "prefix": str,
    "path": str,
})


def create_redis_config(
        host: str,
        port: int,
        passwd: str,
        prefix: str,
        path: str) -> RedisConfig:
    return {
        "host": host,
        "port": port,
        "passwd": passwd,
        "prefix": prefix,
        "path": path,
    }


def get_test_config() -> RedisConfig:
    return {
        "host": "localhost",
        "port": 6380,
        "passwd": "",
        "prefix": "",
        "path": os.path.abspath("test"),
    }


def get_api_config() -> RedisConfig:
    base_path = envload_path("USER_PATH", default="userdata")
    return {
        "host": envload_str("API_REDIS_HOST", default="localhost"),
        "port": envload_int("API_REDIS_PORT", default=6379),
        "passwd": envload_str("API_REDIS_PASSWD", default=""),
        "prefix": envload_str("API_REDIS_PREFIX", default=""),
        "path": os.path.abspath(os.path.join(base_path, "_api")),
    }


RedisSlowMode = Literal["once", "always", "never"]


REDIS_SLOW_MODE: RedisSlowMode = "once"


def set_redis_slow_mode(mode: RedisSlowMode) -> None:
    global REDIS_SLOW_MODE

    REDIS_SLOW_MODE = mode


REDIS_SLOW = 1.0
REDIS_SLOW_CONTEXT = 3
REDIS_UNIQUE: set[tuple[str, int, str]] = set()


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
    "embed",
    "link",
    "locks",
    "test",
    "token",
]
LOCK_MODULE: RedisModule = "locks"


ConfigKey = tuple[str, str]
ConnectionKey = tuple[ConfigKey, str, str]


SCRIPT_CACHE: dict[str, str] = {}
SCRIPT_UTILS_CACHE: str = ""
SCRIPT_UTILS_LINES: int = 0
CONCURRENT_MODULE_CONN: int = 17
REDIS_CONFIG_CACHE: dict[ConfigKey, RedisConfig] = {}
REDIS_SERVICE_CONN: dict[ConnectionKey, StrictRedis | None] = {}
DO_CACHE = True
LOCK = threading.RLock()
LOCKS: dict[str, tuple[threading.RLock, Lock] | None] = {}
LOCK_ID: str = uuid.uuid4().hex


def get_redis_ns_key(ns_name: str, ns_module: str) -> ConfigKey:
    return f"{ns_name}", f"{ns_module}"


def get_connection_key(
        ns_key: ConfigKey, redis_module: RedisModule) -> ConnectionKey:
    return (
        ns_key,
        f"{redis_module}",
        f"{threading.get_ident() % CONCURRENT_MODULE_CONN}",
    )


def register_redis_ns(ns_key: ConfigKey, cfg: RedisConfig) -> None:
    if ns_key[0].startswith("_"):
        raise ValueError(f"invalid ns_key: {ns_key}")
    if ns_key in REDIS_CONFIG_CACHE:
        raise ValueError(f"redis ns already registered: {ns_key}")
    REDIS_CONFIG_CACHE[ns_key] = cfg


REDIS_TEST_CONFIG: ConfigKey = ("_test", "")
REDIS_API_CONFIG: ConfigKey = ("_api", "")


def get_redis_config(ns_key: ConfigKey) -> RedisConfig:
    if ns_key[0] == REDIS_API_CONFIG[0]:
        return get_api_config()
    if ns_key[0] == REDIS_TEST_CONFIG[0]:
        return get_test_config()
    return REDIS_CONFIG_CACHE[ns_key]


class RedisWrapper:
    def __init__(
            self,
            ns_key: ConfigKey,
            redis_module: RedisModule) -> None:
        self._ns_key = ns_key
        self._redis_module = redis_module
        self._conn = None

    def get_ns_key(self) -> ConfigKey:
        return self._ns_key

    @staticmethod
    def _create_connection(cfg: RedisConfig) -> StrictRedis:
        config = {
            "retry_on_timeout": True,
            "health_check_interval": 45,
            "client_name": f"api-{uuid.uuid4().hex}",
        }
        return StrictRedis(  # pylint: disable=unexpected-keyword-arg
            host=cfg["host"],
            port=cfg["port"],
            db=0,
            password=cfg["passwd"],
            **config)

    @classmethod
    def _get_redis_cached_conn(
            cls,
            ns_key: ConfigKey,
            redis_module: RedisModule) -> StrictRedis:
        cfg = get_redis_config(ns_key)
        if not DO_CACHE:
            return cls._create_connection(cfg)

        key = get_connection_key(ns_key, redis_module)
        res = REDIS_SERVICE_CONN.get(key)
        if res is None:
            with LOCK:
                res = REDIS_SERVICE_CONN.get(key)
                if res is None:
                    res = cls._create_connection(cfg)
                    REDIS_SERVICE_CONN[key] = res
        return res

    @classmethod
    def _get_lock_pair(
            cls, ns_key: ConfigKey, name: str) -> tuple[threading.RLock, Lock]:
        res = LOCKS.get(name)
        if res is None:
            with LOCK:
                res = LOCKS.get(name)
                if res is None:
                    res = (
                        threading.RLock(),
                        Lock(
                            cls._get_redis_cached_conn(ns_key, LOCK_MODULE),
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
                self._conn = self._get_redis_cached_conn(
                    self._ns_key, self._redis_module)
            yield self._conn
        except Exception:
            self.reset()
            raise
        finally:
            conn_time = time.monotonic() - conn_start
            if conn_time > REDIS_SLOW and REDIS_SLOW_MODE != "never":
                fun_info = get_relative_function_info(depth=depth + 1)
                fun_key = fun_info[:3]
                if fun_key not in REDIS_UNIQUE or REDIS_SLOW_MODE == "always":
                    fun_fname, fun_line, fun_name, fun_locals = fun_info
                    context = []
                    try:
                        fun_line -= 1
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
                    except (OSError, FileNotFoundError):
                        context.append("## not available ##")
                    print(
                        f"slow redis call ({conn_time:.2f}s) "
                        f"at {fun_name} ({fun_fname}:{fun_line})\n"
                        f"{NL.join(context)}\nlocals:\n{fun_locals}\n"
                        f"conn:{json_pretty(get_redis_config(self._ns_key))}")
                    REDIS_UNIQUE.add(fun_key)

    def reset(self) -> None:
        conn = self._conn
        if conn is not None:
            self._invalidate_connection(self._ns_key, self._redis_module)
            self._conn = None

    @classmethod
    def reset_lock(cls, ns_key: ConfigKey) -> None:
        with LOCK:
            LOCKS.clear()
            cls._invalidate_connection(ns_key, LOCK_MODULE)

    @staticmethod
    def _invalidate_connection(
            _ns_key: ConfigKey, _redis_module: RedisModule) -> None:
        with LOCK:
            # key = get_connection_key(ns_key, redis_module)
            # REDIS_SERVICE_CONN[key] = None
            # NOTE: prevents issues from coming up multiple times
            REDIS_SERVICE_CONN.clear()

    @classmethod
    @contextlib.contextmanager
    def create_lock(cls, ns_key: ConfigKey, name: str) -> Iterator[Lock]:
        try:
            local_lock, redis_lock = cls._get_lock_pair(ns_key, name)
            with local_lock:
                yield redis_lock
        except Exception:
            cls.reset_lock(ns_key)
            raise

    @staticmethod
    @contextlib.contextmanager
    def no_lock() -> Iterator[None]:
        yield


class RedisConnection:
    def __init__(self, ns_key: ConfigKey, redis_module: RedisModule) -> None:
        self._conn = RedisWrapper(ns_key, redis_module)
        cfg = get_redis_config(ns_key)
        salt = get_test_salt()
        salt_str = "" if salt is None else f"{salt}:"
        prefix_str = f"{cfg['prefix']}:" if cfg["prefix"] else ""
        self._module = f"{salt_str}{prefix_str}{redis_module}"

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
            self, prefix: str, postfix: str | None = None) -> Iterable[str]:
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
                if count < 1000:
                    count = int(min(1000, count * 1.2))
        return (val.decode("utf-8") for val in vals)

    def keys_gap_str(
            self,
            prefix: str,
            gap: int,
            postfix: str | None = None) -> Iterable[str]:
        glob = "*" if postfix is None else "?" * gap
        full_pattern = f"{prefix}{glob}{'' if postfix is None else postfix}"
        vals: set[bytes] = set()
        cursor = 0
        count = 10
        with self.get_connection(depth=1) as conn:
            while True:
                cursor, res = conn.scan(cursor, full_pattern, count)
                vals.update(res)
                if cursor == 0:
                    break
                if count < 1000:
                    count = int(min(1000, count * 1.2))
        return (val.decode("utf-8") for val in vals)

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
                if count < 1000:
                    count = int(min(1000, count * 1.2))

    @contextlib.contextmanager
    def get_lock(self, name: str) -> Iterator[None]:
        with self._conn.create_lock(
                self._conn.get_ns_key(), f"{self.get_prefix()}:{name}"):
            yield


class ObjectRedis(RedisConnection):
    def __init__(self, ns_key: ConfigKey, redis_module: RedisModule) -> None:
        super().__init__(ns_key, redis_module)
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

    def obj_keys(self, name: str) -> Iterable[str]:
        path = self.compute_name(name, key="")
        return (key[len(path):] for key in self.keys_str(path))

    def obj_dict(self, name: str) -> dict[str, Any]:
        path = self.compute_name(name, key="")
        keys = list(self.keys_str(path))
        with self.get_connection(depth=1) as conn:
            return {
                key[len(path):]: json_read(res)
                for (key, res) in zip(keys, conn.mget(keys), strict=True)
                if res is not None
            }

    def obj_keys_count(self, name: str) -> int:
        path = self.compute_name(name, key="")
        return self.keys_count(path)

    def obj_partial_keys(self, partial: str) -> Iterable[str]:
        path = f"{self._objs}:{partial}"
        return (key[len(path):] for key in self.keys_str(path))

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
