import re
from typing import Any, cast, get_args, Optional, TypedDict


VALID_NAME = re.compile(r"^\W+$")


Permissions = TypedDict('Permissions', {
    "can_create_topic": bool,
})
PERMISSIONS_KEYS = get_args(Permissions)


def ensure_permissions(obj: Any) -> Permissions:
    if len(obj.keys() | set(PERMISSIONS_KEYS)) != len(PERMISSIONS_KEYS):
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
        self._user_id: Optional[str] = None
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
            owner: Optional['User'],  # pylint: disable=unused-argument
            ) -> float:
        # FIXME: implement ELO
        return 1.0

    def __hash__(self) -> int:
        return hash(self.get_name())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return self.get_name() == other.get_name()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.get_name()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.get_name()}]"
