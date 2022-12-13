from typing import Iterable, Literal, TypedDict

from misc.util import get_short_hash
from system.namespace.namespace import Namespace
from system.users.user import User


class UserStore:
    def get_user_by_id(self, user_id: str) -> User:
        raise NotImplementedError()

    def store_user(self, user: User) -> None:
        raise NotImplementedError()

    def get_all_users(self) -> Iterable[User]:
        raise NotImplementedError()

    @staticmethod
    def get_id_from_name(user_name: str) -> str:
        return get_short_hash(user_name)


USER_STORE: dict[Namespace, UserStore] = {}


def get_user_store(namespace: Namespace) -> UserStore:
    res = USER_STORE.get(namespace)
    if res is None:
        res = create_user_store(namespace.get_users_module())
        USER_STORE[namespace] = res
    return res


UsersStoreName = Literal["disk", "ram"]


UsersModule = TypedDict('UsersModule', {
    "name": UsersStoreName,
    "folder": str,
})


def create_user_store(uobj: UsersModule) -> UserStore:
    name = uobj["name"]
    if name == "disk":
        from system.users.disk import DiskUserStore
        return DiskUserStore()
    if name == "ram":
        from system.users.ram import RamUserStore
        return RamUserStore()
    raise ValueError(f"unknown user store: {name}")
