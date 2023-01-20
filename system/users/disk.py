import os
from typing import Callable, Iterable

from misc.io import ensure_folder, open_append, open_read
from misc.lru import LRU
from misc.util import json_compact, read_jsonl
from system.users.store import UserStore
from system.users.user import User


USER_EXT = ".user"


class DiskUserStore(UserStore):
    def __init__(self, user_root: str, cache_size: int) -> None:
        super().__init__()
        self._path = os.path.join(user_root, "users")
        self._cache: LRU[str, User] = LRU(cache_size)

    def _compute_path(self, user_id: str) -> str:
        # FIXME: create generic class with dedup and subtree creation

        def split_hash(hash_str: str) -> Iterable[str]:
            yield hash_str[:2]
            yield hash_str[2:4]
            # NOTE: we ignore the last segment
            # yield hash_str[4:]

        all_segs = list(split_hash(user_id))
        segs = all_segs[:-1]
        rest = all_segs[-1]
        return os.path.join(self._path, *segs, f"{rest}{USER_EXT}")

    def store_user(self, user: User) -> None:
        user_id = user.get_id()
        ensure_folder(os.path.dirname(self._compute_path(user_id)))
        with open_append(self._compute_path(user_id), text=True) as fout:
            fout.write(
                f"{json_compact(self.get_user_dict(user)).decode('utf-8')}\n")

    def _get_users_for_file(self, fname: str) -> Iterable[User]:
        users = set()
        try:
            with open_read(fname, text=True) as fin:
                for obj in read_jsonl(fin):
                    uobj = self.ensure_user_dict(obj)
                    user = User(uobj["name"], uobj["permissions"])
                    users.add(user)
        except FileNotFoundError:
            pass
        yield from users

    def get_user_by_id(self, user_id: str) -> User:
        res = self._cache.get(user_id)
        if res is not None:
            return res
        fname = self._compute_path(user_id)
        for user in self._get_users_for_file(fname):
            self._cache.set(user.get_id(), user)
            if user.get_id() == user_id:
                res = user
        if res is None:
            raise KeyError(f"no user for the id: {user_id}")
        return res

    def get_all_users(self, *, progress_bar: bool) -> Iterable[User]:
        all_files = list(os.walk(self._path))

        def get_results(*, pbar: Callable[[], None] | None) -> Iterable[User]:
            for (root, _, files) in all_files:
                for fname in files:
                    if fname.endswith(USER_EXT):
                        yield from self._get_users_for_file(
                            os.path.join(root, fname))
                    if pbar is not None:
                        pbar()

        if not progress_bar:
            yield from get_results(pbar=None)
        else:
            # FIXME: add stubs
            from tqdm.auto import tqdm  # type: ignore

            with tqdm(total=len(all_files)) as pbar:
                yield from get_results(pbar=lambda: pbar.update(1))
