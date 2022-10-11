from typing import Dict, Iterable

from system.users.store import UserStore
from system.users.user import User


class RamUserStore(UserStore):
    def __init__(self) -> None:
        self._users: Dict[str, User] = {}

    def get_user_by_id(self, user_id: str) -> User:
        return self._users[user_id]

    def store_user(self, user: User) -> None:
        self._users[user.get_id()] = user

    def get_all_users(self) -> Iterable[User]:
        yield from self._users.values()
