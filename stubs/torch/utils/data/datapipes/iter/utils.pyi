from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class IterableWrapperIterDataPipe(IterDataPipe):
    iterable: Incomplete
    deepcopy: Incomplete
    def __init__(self, iterable, deepcopy: bool = ...) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...
