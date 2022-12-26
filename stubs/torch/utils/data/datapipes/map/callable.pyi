from typing import Callable, TypeVar

from torch.utils.data.datapipes.datapipe import MapDataPipe


T_co = TypeVar('T_co', covariant=True)

def default_fn(data): ...

class MapperMapDataPipe(MapDataPipe[T_co]):
    datapipe: MapDataPipe
    fn: Callable
    def __init__(self, datapipe: MapDataPipe, fn: Callable = ...) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index) -> T_co: ...
