from typing import Tuple

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class StreamReaderIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    datapipe: Incomplete
    chunk: Incomplete
    def __init__(self, datapipe, chunk: Incomplete | None = ...) -> None: ...
    def __iter__(self): ...
