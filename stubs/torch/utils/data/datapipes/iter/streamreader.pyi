# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from typing import Tuple

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class StreamReaderIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    datapipe: Incomplete
    chunk: Incomplete
    def __init__(self, datapipe, chunk: Incomplete | None = ...) -> None: ...
    def __iter__(self): ...
