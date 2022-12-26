from _typeshed import Incomplete

from .glob_group import GlobGroup as GlobGroup
from .glob_group import GlobPattern as GlobPattern


class Directory:
    name: Incomplete
    is_dir: Incomplete
    children: Incomplete
    def __init__(self, name: str, is_dir: bool) -> None: ...
    def has_file(self, filename: str) -> bool: ...
