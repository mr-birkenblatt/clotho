# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe


T = TypeVar('T')


class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool
    length: Optional[int]
    wrapper_class: Incomplete

    def __init__(
        self, datapipe: MapDataPipe[T], batch_size: int,
        drop_last: bool = ..., wrapper_class=...) -> None: ...

    def __getitem__(self, index) -> DataChunk: ...
    def __len__(self) -> int: ...
