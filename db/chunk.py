import os
from typing import Iterable

from db.lsm import ChunkCoordinator, KeyRange, LSM, VType
from misc.io import listdir


FILE_EXT = ".lsm"


class DiskChunk(ChunkCoordinator):
    def __init__(self, prefix: str, root: str) -> None:
        self._prefix = prefix
        self._root = root
        self._children: dict[str, DiskChunk] | None = None
        self._files: list[str] | None = None

    def load_dir(self) -> None:
        children: dict[str, DiskChunk] = {}
        files: list[str] = []
        for fname in listdir(self._root):
            full = os.path.join(self._root, fname)
            if os.path.isdir(full):
                children[fname] = DiskChunk(f"{self._prefix}{fname}", full)
            elif full.endswith(FILE_EXT):
                files.append(full)
        self._children = children
        self._files = files

    def read_file(self, full: str) -> tuple[float, dict[str, VType]]:
        raise NotImplementedError()

    def write_values(
            self,
            cur_time: float,
            values: dict[str, VType],
            lsm: LSM) -> None:
        raise NotImplementedError()

    def fetch_values(
            self, key: str, lsm: LSM) -> Iterable[tuple[str, VType]]:
        raise NotImplementedError()

    def clear_cache(self, key_range: KeyRange) -> None:
        pass
