import os
from typing import Dict, Iterable, List, Optional, Tuple

from db.lsm import ChunkCoordinator, KeyRange, LSM, VType


FILE_EXT = ".lsm"


class DiskChunk(ChunkCoordinator):
    def __init__(self, prefix: str, root: str) -> None:
        self._prefix = prefix
        self._root = root
        self._children: Optional[Dict[str, DiskChunk]] = None
        self._files: Optional[List[str]] = None

    def load_dir(self) -> None:
        children: Dict[str, DiskChunk] = {}
        files: List[str] = []
        for fname in os.listdir(self._root):
            full = os.path.join(self._root, fname)
            if os.path.isdir(full):
                children[fname] = DiskChunk(f"{self._prefix}{fname}", full)
            elif full.endswith(FILE_EXT):
                files.append(full)
        self._children = children
        self._files = files

    def read_file(self, full: str) -> Tuple[float, Dict[str, VType]]:
        pass

    def write_values(
            self,
            cur_time: float,
            values: Dict[str, VType],
            lsm: LSM) -> None:
        raise NotImplementedError()

    def fetch_values(
            self, key: str, lsm: LSM) -> Iterable[Tuple[str, VType]]:
        raise NotImplementedError()

    def clear_cache(self, key_range: KeyRange) -> None:
        pass
