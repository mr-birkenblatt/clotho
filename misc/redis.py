import contextlib
import os
import threading
import uuid
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    overload,
    Protocol,
    Set,
    Tuple,
    Union,
)

from redis import StrictRedis
from redis.client import Script
from redis.exceptions import ResponseError
from redis_lock import Lock

from misc.env import envload_int, envload_str
from misc.util import json_compact, json_read
from misc.io import open_read


class RedisFunctionBytes(Protocol):  # pylint: disable=too-few-public-methods
    def __call__(
            self,
            keys: List[str],
            args: List[Any],
            client: Optional[StrictRedis] = None) -> bytes:
        ...


class RedisFunctionList(Protocol):  # pylint: disable=too-few-public-methods
    def __call__(
            self,
            keys: List[str],
            args: List[Any],
            client: Optional[StrictRedis] = None) -> List[bytes]:
        ...


RedisModule = Literal[
    "link",
    "locks",
    "token",
]
LOCK_MODULE: RedisModule = "locks"


SCRIPT_CACHE: Dict[str, str] = {}
SCRIPT_UTILS_CACHE: str = ""
SCRIPT_UTILS_LINES: int = 0
REDIS_SERVICE_CONN: Dict[RedisModule, Optional[StrictRedis]] = {}
DO_CACHE = True
LOCK = threading.RLock()
LOCKS: Dict[str, Optional[Tuple[threading.RLock, Lock]]] = {}
LOCK_ID: str = str(uuid.uuid4())


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

        res = REDIS_SERVICE_CONN.get(module)
        if res is None:
            with LOCK:
                res = REDIS_SERVICE_CONN.get(module)
                if res is None:
                    res = cls._create_connection()
                    REDIS_SERVICE_CONN[module] = res
        return res

    @classmethod
    def _get_lock_pair(cls, name: str) -> Tuple[threading.RLock, Lock]:
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
    def get_connection(self) -> Iterator[StrictRedis]:
        try:
            if self._conn is None:
                self._conn = self._get_redis_cached_conn(self._module)
            yield self._conn
        except Exception:
            self.reset()
            raise

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
            REDIS_SERVICE_CONN[module] = None

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
        self._module = module

    def get_connection(self) -> ContextManager[StrictRedis]:
        return self._conn.get_connection()

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
            return_list: bool) -> Union[
                RedisFunctionBytes, RedisFunctionList]:
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

        def get_error(err_msg: str) -> Optional[Tuple[str, List[str]]]:
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
        def get_client(client: Optional[StrictRedis]) -> Iterator[StrictRedis]:
            try:
                if client is None:
                    with self.get_connection() as res:
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
                keys: List[str],
                args: List[Union[bytes, str, int]],
                client: Optional[StrictRedis] = None) -> List[bytes]:
            with get_client(client) as inner:
                res = compute(keys=keys, args=args, client=inner)
                assert isinstance(res, list)
                return res

        def execute_bytes_result(
                keys: List[str],
                args: List[Union[bytes, str, int]],
                client: Optional[StrictRedis] = None) -> bytes:
            with get_client(client) as inner:
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
        with self.get_connection() as conn:
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
        with self.get_connection() as conn:
            conn.publish(self.get_pubsub_key(key), "notify")

    def ping(self) -> None:
        with self.get_connection() as conn:
            conn.ping()

    def flush_all(self) -> None:
        with self.get_connection() as conn:
            conn.flushall()

    def keys_count(self, prefix: str) -> int:
        full_prefix = f"{prefix}*"
        vals: Set[bytes] = set()
        cursor = 0
        count = 10
        with self.get_connection() as conn:
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

    def keys_str(self, prefix: str) -> Iterable[str]:
        full_prefix = f"{prefix}*"
        vals: Set[bytes] = set()
        cursor = 0
        count = 10
        with self.get_connection() as conn:
            while True:
                cursor, res = conn.scan(cursor, full_prefix, count)
                vals.update(res)
                if cursor == 0:
                    break
                if count < 4000:
                    count = min(4000, count * 2)
        return (val.decode("utf-8") for val in vals)

    def prefix_exists(self, prefix: str) -> bool:
        full_prefix = f"{prefix}*"
        cursor = 0
        count = 10
        with self.get_connection() as conn:
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

    def obj_flush_keys(self, patterns: List[str], skip: List[str]) -> None:
        self._flush_keys(keys=[], args=[json_compact(patterns), *skip])

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
            args=[int(preserve_expire), json_compact(value)])

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
        with self.get_connection() as conn:
            res = int(conn.setnx(path, value))
        return res != 0

    def obj_put_expire(
            self,
            name: str,
            key: str,
            value: Any,
            expire: Optional[float]) -> None:
        path = self.compute_name(name, key)
        with self.get_connection() as conn:
            conn.set(
                path,
                json_compact(value),
                px=None if expire is None else int(expire * 1000))

    def obj_expire(self, name: str, key: str, expire: float) -> None:
        with self.get_connection() as conn:
            conn.pexpire(self.compute_name(name, key), int(expire * 1000))

    def obj_get(self, name: str, key: str, default: Any = None) -> Any:
        with self.get_connection() as conn:
            res = conn.get(self.compute_name(name, key))
        return json_read(res) if res is not None else default

    def obj_get_expire(
            self,
            name: str,
            key: str,
            expire: Optional[float],
            default: Any = None) -> Any:
        path = self.compute_name(name, key)
        with self.get_connection() as conn:
            # expire before reading to ensure correct treatment of 0
            if expire is not None:
                conn.pexpire(path, int(expire * 1000))
            res = conn.get(path)
        return json_read(res) if res is not None else default

    def obj_ttl(self, name: str, key: str) -> Optional[float]:
        # pylint: disable=unbalanced-tuple-unpacking

        path = self.compute_name(name, key)
        with self.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.exists(path)
                pipe.pttl(path)
                exists, pexpire = pipe.execute()
        if exists:
            return max(pexpire / 1000.0, 0.0)
        return None

    def obj_has(self, name: str, key: str) -> bool:
        with self.get_connection() as conn:
            return conn.get(self.compute_name(name, key)) is not None

    def obj_keys(self, name: str) -> List[str]:
        path = self.compute_name(name, key="")
        return [
            key[len(path):]
            for key in self.keys_str(path)
        ]

    def obj_dict(self, name: str) -> Dict[str, Any]:
        path = self.compute_name(name, key="")
        keys = list(self.keys_str(path))
        with self.get_connection() as conn:
            return {
                key[len(path):]: json_read(res)
                for (key, res) in zip(keys, conn.mget(keys))
                if res is not None
            }

    def obj_keys_count(self, name: str) -> int:
        path = self.compute_name(name, key="")
        return self.keys_count(path)

    def obj_partial_keys(self, partial: str) -> List[str]:
        path = f"{self._objs}:{partial}"
        return [
            key[len(path):]
            for key in self.keys_str(path)
        ]

    def obj_remove(self, name: str, key: str = "*") -> None:
        with self.get_connection() as conn:
            conn.delete(self.compute_name(name, key))

    def obj_remove_all(self, name: str, keys: List[str]) -> None:
        with self.get_connection() as conn:
            for key in keys:
                conn.delete(self.compute_name(name, key))

    def obj_pop_raw(self, name: str, key: str) -> Optional[bytes]:
        path = self.compute_name(name, key)
        with self.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.get(path)
                pipe.delete(path)
                return pipe.execute()[0]

    def obj_pop(self, name: str, key: str, default: Any = None) -> Any:
        path = self.compute_name(name, key)
        with self.get_connection() as conn:
            with conn.pipeline() as pipe:
                pipe.get(path)
                pipe.delete(path)
                res = pipe.execute()[0]
        return json_read(res) if res is not None else default
