from typing import Iterable

from system.users.store import UserStore
from system.users.user import User


class RamUserStore(UserStore):
    def __init__(self) -> None:
        super().__init__()
        self._users: dict[str, User] = {}

    def is_module_init(self) -> bool:
        return True

    def get_user_by_id(self, user_id: str) -> User:
        return self._users[user_id]

    def store_user(self, user: User) -> None:
        self._users[user.get_id()] = user

    def get_all_users(self) -> Iterable[User]:
        yield from self._users.values()
