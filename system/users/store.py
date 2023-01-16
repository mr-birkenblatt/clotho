from typing import Iterable, Literal, TypedDict

from misc.util import get_short_hash
from system.namespace.module import ModuleBase
from system.namespace.namespace import Namespace
from system.users.user import User


class UserStore(ModuleBase):
    @staticmethod
    def module_name() -> str:
        return "users"

    def get_user_by_id(self, user_id: str) -> User:
        raise NotImplementedError()

    def store_user(self, user: User) -> None:
        raise NotImplementedError()

    def get_all_users(self) -> Iterable[User]:
        raise NotImplementedError()

    def from_namespace(
            self, other_namespace: Namespace, *, progress_bar: bool) -> None:
        ousers = get_user_store(other_namespace)
        for user in ousers.get_all_users():
            self.store_user(user)

    @staticmethod
    def get_id_from_name(user_name: str) -> str:
        return get_short_hash(user_name)


USER_STORE: dict[Namespace, UserStore] = {}


def get_user_store(namespace: Namespace) -> UserStore:
    res = USER_STORE.get(namespace)
    if res is None:
        res = create_user_store(namespace)
        USER_STORE[namespace] = res
    return res


DiskUsersModule = TypedDict('DiskUsersModule', {
    "name": Literal["disk"],
    "cache_size": int,
})
RamUsersModule = TypedDict('RamUsersModule', {
    "name": Literal["ram"],
})
UsersModule = DiskUsersModule | RamUsersModule


def create_user_store(namespace: Namespace) -> UserStore:
    uobj = namespace.get_users_module()
    if uobj["name"] == "disk":
        from system.users.disk import DiskUserStore
        return DiskUserStore(namespace.get_root(), uobj["cache_size"])
    if uobj["name"] == "ram":
        from system.users.ram import RamUserStore
        return RamUserStore()
    raise ValueError(f"unknown user store: {uobj}")
