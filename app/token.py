from typing import Any, ContextManager

from quick_server import TokenHandler

from misc.redis import ObjectRedis, REDIS_API_CONFIG, RedisWrapper


class RedisTokenHandler(TokenHandler):
    def __init__(self) -> None:
        self._r = ObjectRedis(REDIS_API_CONFIG, "token")

    def lock(self, key: str | None) -> ContextManager[None]:
        if key is None:
            return RedisWrapper.no_lock()
        return RedisWrapper.create_lock(REDIS_API_CONFIG, f"token.{key}")

    def ttl(self, key: str) -> float | None:
        return self._r.obj_ttl("token", key)

    def flush_old_tokens(self) -> None:
        pass  # we don't need to manually remove tokens

    def add_token(self, key: str, expire: float | None) -> dict[str, Any]:
        res = self._r.obj_get_expire("token", key, expire)
        if res is None:
            res = {}
            if expire is None or expire > 0:
                self._r.obj_put_expire("token", key, res, expire)
        return res

    def put_token(self, key: str, obj: dict[str, Any]) -> None:
        self._r.obj_put("token", key, obj, preserve_expire=True)

    def delete_token(self, key: str) -> None:
        self._r.obj_remove("token", key)

    def get_tokens(self) -> list[str]:
        return list(self._r.obj_keys("token"))
