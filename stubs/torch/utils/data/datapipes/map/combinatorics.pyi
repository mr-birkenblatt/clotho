# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from typing import Iterator, List, Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import MapDataPipe


T_co = TypeVar('T_co', covariant=True)


class ShufflerMapDataPipe(MapDataPipe[T_co]):
    datapipe: MapDataPipe[T_co]
    indices: Incomplete
    index_map: Incomplete

    def __init__(
        self,
        datapipe: MapDataPipe[T_co],
        *,
        indices: Optional[List] = ...) -> None: ...

    def __getitem__(self, index) -> T_co: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...
