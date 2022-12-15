import os
from typing import Any, Iterable, TypedDict

from misc.env import envload_path
from misc.io import ensure_folder, open_append, open_read
from misc.lru import LRU
from misc.util import json_compact, read_jsonl
from system.users.store import UserStore
from system.users.user import ensure_permissions, Permissions, User


USER_EXT = ".user"


UserDict = TypedDict('UserDict', {
    "name": str,
    "permissions": Permissions,
})


def ensure_user_dict(obj: Any) -> UserDict:
    return {
        "name": obj["name"],
        "permissions": ensure_permissions(obj["permissions"]),
    }


class DiskUserStore(UserStore):
    def __init__(self, user_root: str) -> None:
        base_path = envload_path("USER_PATH", default="userdata")
        self._path = os.path.join(base_path, user_root, "user")
        self._cache: LRU[str, User] = LRU(10000)

    def _get_user_dict(self, user: User) -> UserDict:
        return {
            "name": user.get_name(),
            "permissions": user.get_permissions(),
        }

    def _compute_path(self, user_id: str) -> str:
        # FIXME: create generic class with dedup and subtree creation

        def split_hash(hash_str: str) -> Iterable[str]:
            yield hash_str[:2]
            yield hash_str[2:4]
            # NOTE: we ignore the last segment
            # yield hash_str[4:]

        all_segs = list(split_hash(user_id))
        segs = all_segs[:-1]
        rest = all_segs[-1]
        return os.path.join(self._path, *segs, f"{rest}{USER_EXT}")

    def store_user(self, user: User) -> None:
        user_id = user.get_id()
        ensure_folder(os.path.dirname(self._compute_path(user_id)))
        with open_append(self._compute_path(user_id), text=True) as fout:
            fout.write(
                f"{json_compact(self._get_user_dict(user)).decode('utf-8')}\n")

    def _get_users_for_file(self, fname: str) -> Iterable[User]:
        users = set()
        try:
            with open_read(fname, text=True) as fin:
                for obj in read_jsonl(fin):
                    uobj = ensure_user_dict(obj)
                    user = User(uobj["name"], uobj["permissions"])
                    users.add(user)
        except FileNotFoundError:
            pass
        yield from users

    def get_user_by_id(self, user_id: str) -> User:
        res = self._cache.get(user_id)
        if res is not None:
            return res
        fname = self._compute_path(user_id)
        for user in self._get_users_for_file(fname):
            self._cache.set(user.get_id(), user)
            if user.get_id() == user_id:
                res = user
        if res is None:
            raise KeyError(f"no user for the id: {user_id}")
        return res

    def get_all_users(self) -> Iterable[User]:
        for (root, _, files) in os.walk(self._path):
            for fname in files:
                if not fname.endswith(USER_EXT):
                    continue
                yield from self._get_users_for_file(os.path.join(root, fname))
