from io import IOBase
from typing import Iterable, Optional, Tuple

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class FileOpenerIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    datapipe: Incomplete
    mode: Incomplete
    encoding: Incomplete
    length: Incomplete
    def __init__(self, datapipe: Iterable[str], mode: str = ..., encoding: Optional[str] = ..., length: int = ...) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...

class FileLoaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    def __new__(cls, datapipe: Iterable[str], mode: str = ..., length: int = ...): ...
