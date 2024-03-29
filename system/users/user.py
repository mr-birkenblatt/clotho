import re
from typing import Any, cast, TypedDict


VALID_NAME = re.compile(r"^\S+$")
MAX_USER_NAME_LEN = 100


Permissions = TypedDict('Permissions', {
    "can_create_topic": bool,
})
PERMISSIONS_KEYS = {
    "can_create_topic",
}


def ensure_permissions(obj: Any) -> Permissions:
    if len(obj.keys() | PERMISSIONS_KEYS) != len(PERMISSIONS_KEYS):
        raise ValueError(f"wrong keys: {obj.keys()} != {PERMISSIONS_KEYS}")
    return cast(Permissions, {
        key: bool(obj[key])
        for key in PERMISSIONS_KEYS
    })


class User:
    def __init__(
            self,
            name: str,
            permissions: Permissions) -> None:
        if VALID_NAME.search(name) is None:
            raise ValueError(f"invalid user name {name}")
        if len(name) > MAX_USER_NAME_LEN:
            raise ValueError(f"user name too long: {name}")
        self._user_id: str | None = None
        self._name = name
        self._permissions = permissions

    def get_id(self) -> str:
        from system.users.store import UserStore

        res = self._user_id
        if res is None:
            res = UserStore.get_id_from_name(self._name)
            self._user_id = res
        return res

    def get_name(self) -> str:
        return self._name

    def can_create_topic(self) -> bool:
        return self._permissions.get("can_create_topic", False)

    def get_permissions(self) -> Permissions:
        return self._permissions.copy()

    def get_reputation(self) -> float:
        # FIXME implement ELO
        return 1000.0

    def get_encounters(
            self, other: 'User') -> int:  # pylint: disable=unused-argument
        # FIXME: implement encounters
        return 0

    def get_weighted_vote(
            self,
            owner: 'User | None') -> float:  # pylint: disable=unused-argument
        # FIXME: implement ELO
        return 1.0

    def __hash__(self) -> int:
        return hash(self.get_id())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return self.get_id() == other.get_id()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.get_name()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.get_name()}]"
