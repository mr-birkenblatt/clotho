from typing import Any, Iterable, Literal, TypedDict

from misc.util import get_short_hash
from system.namespace.module import ModuleBase
from system.namespace.namespace import ModuleName, Namespace
from system.users.user import ensure_permissions, Permissions, User


UserDict = TypedDict('UserDict', {
    "name": str,
    "permissions": Permissions,
})


class UserStore(ModuleBase):
    @staticmethod
    def module_name() -> ModuleName:
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

    def get_user_dict(self, user: User) -> UserDict:
        return {
            "name": user.get_name(),
            "permissions": user.get_permissions(),
        }

    def ensure_user_dict(self, obj: Any) -> UserDict:
        return {
            "name": obj["name"],
            "permissions": ensure_permissions(obj["permissions"]),
        }

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
ColdUsersModule = TypedDict('ColdUsersModule', {
    "name": Literal["cold"],
    "keep_alive": float,
})
UsersModule = DiskUsersModule | RamUsersModule | ColdUsersModule


def create_user_store(namespace: Namespace) -> UserStore:
    uobj = namespace.get_users_module()
    if uobj["name"] == "disk":
        from system.users.disk import DiskUserStore
        return DiskUserStore(namespace.get_root(), uobj["cache_size"])
    if uobj["name"] == "ram":
        from system.users.ram import RamUserStore
        return RamUserStore()
    if uobj["name"] == "cold":
        from system.users.cold import ColdUserStore
        return ColdUserStore(
            namespace.get_module_root("users"), keep_alive=uobj["keep_alive"])
    raise ValueError(f"unknown user store: {uobj}")
