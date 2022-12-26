from typing import Tuple, TypeVar

from torch.utils.data.datapipes.datapipe import MapDataPipe


T_co = TypeVar('T_co', covariant=True)


class ConcaterMapDataPipe(MapDataPipe):
    datapipes: Tuple[MapDataPipe]
    length: int
    def __init__(self, *datapipes: MapDataPipe) -> None: ...
    def __getitem__(self, index) -> T_co: ...
    def __len__(self) -> int: ...


class ZipperMapDataPipe(MapDataPipe[Tuple[T_co, ...]]):
    datapipes: Tuple[MapDataPipe[T_co], ...]
    length: int
    def __init__(self, *datapipes: MapDataPipe[T_co]) -> None: ...
    def __getitem__(self, index) -> Tuple[T_co, ...]: ...
    def __len__(self) -> int: ...
