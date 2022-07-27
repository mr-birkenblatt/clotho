from typing import Any, ContextManager, Dict, List, Optional
from misc.redis import ObjectRedis, RedisWrapper
from quick_server import TokenHandler


class RedisTokenHandler(TokenHandler):
    def __init__(self) -> None:
        self._r = ObjectRedis("token")

    def lock(self, key: Optional[str]) -> ContextManager[None]:
        if key is None:
            return RedisWrapper.no_lock()
        return RedisWrapper.create_lock(f"token.{key}")

    def ttl(self, key: str) -> Optional[float]:
        return self._r.obj_ttl("token", key)

    def flush_old_tokens(self) -> None:
        pass  # we don't need to manually remove tokens

    def add_token(self, key: str, expire: Optional[float]) -> Dict[str, Any]:
        res = self._r.obj_get_expire("token", key, expire)
        if res is None:
            res = {}
            if expire is None or expire > 0:
                self._r.obj_put_expire("token", key, res, expire)
        return res

    def put_token(self, key: str, obj: Dict[str, Any]) -> None:
        self._r.obj_put("token", key, obj, preserve_expire=True)

    def delete_token(self, key: str) -> None:
        self._r.obj_remove("token", key)

    def get_tokens(self) -> List[str]:
        return self._r.obj_keys("token")
