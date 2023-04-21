import os
from typing import Iterable

from misc.cold_writer import ColdAccess
from misc.util import json_compact, json_read
from system.users.store import UserStore
from system.users.user import User


class ColdUserStore(UserStore):
    def __init__(self, root: str, *, keep_alive: float) -> None:
        super().__init__()
        self._out = ColdAccess(
            os.path.join(root, "users.zip"), keep_alive=keep_alive)

    def get_user_by_id(self, user_id: str) -> User:
        raise RuntimeError("cannot random access users in cold storage")

    def store_user(self, user: User) -> None:
        self._out.write_line(
            json_compact(self.get_user_dict(user)).decode("utf-8"))

    def get_all_users(self, *, progress_bar: bool) -> Iterable[User]:
        for line in self._out.enumerate_lines():
            obj = self.ensure_user_dict(json_read(line.encode("utf-8")))
            yield User(obj["name"], obj["permissions"])
