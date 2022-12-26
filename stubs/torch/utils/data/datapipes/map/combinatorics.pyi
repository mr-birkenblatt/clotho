from typing import Iterator, List, Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import MapDataPipe


T_co = TypeVar('T_co', covariant=True)

class ShufflerMapDataPipe(MapDataPipe[T_co]):
    datapipe: MapDataPipe[T_co]
    indices: Incomplete
    index_map: Incomplete
    def __init__(self, datapipe: MapDataPipe[T_co], *, indices: Optional[List] = ...) -> None: ...
    def __getitem__(self, index) -> T_co: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...
