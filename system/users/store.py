from typing import Optional

from misc.env import envload_str
from misc.util import get_short_hash
from system.users.user import User


class UserStore:
    def get_user_by_id(self, user_id: str) -> User:
        raise NotImplementedError()

    def store_user(self, user: User) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_id_from_name(user_name: str) -> str:
        return get_short_hash(user_name)


DEFAULT_USER_STORE: Optional[UserStore] = None


def get_default_user_store() -> UserStore:
    global DEFAULT_USER_STORE

    if DEFAULT_USER_STORE is None:
        DEFAULT_USER_STORE = get_user_store(
            envload_str("USER_STORE", default="disk"))
    return DEFAULT_USER_STORE


def get_user_store(name: str) -> UserStore:
    if name == "disk":
        from system.users.disk import DiskUserStore
        return DiskUserStore()
    raise ValueError(f"unknown user store: {name}")
