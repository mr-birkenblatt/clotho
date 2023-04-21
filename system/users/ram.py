from typing import Iterable

from system.users.store import UserStore
from system.users.user import User


class RamUserStore(UserStore):
    def __init__(self) -> None:
        super().__init__()
        self._users: dict[str, User] = {}

    def get_user_by_id(self, user_id: str) -> User:
        return self._users[user_id]

    def store_user(self, user: User) -> None:
        self._users[user.get_id()] = user

    def get_all_users(self, *, progress_bar: bool) -> Iterable[User]:
        if not progress_bar:
            yield from self._users.values()
            return
        # FIXME: add stubs
        from tqdm.auto import tqdm  # type: ignore

        with tqdm(total=len(self._users)) as pbar:
            for user in list(self._users.values()):
                yield user
                pbar.update(1)
