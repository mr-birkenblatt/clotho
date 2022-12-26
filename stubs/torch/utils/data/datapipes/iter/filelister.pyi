from typing import Iterator, List, Sequence, Union

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class FileListerIterDataPipe(IterDataPipe[str]):
    datapipe: Incomplete
    masks: Incomplete
    recursive: Incomplete
    abspath: Incomplete
    non_deterministic: Incomplete
    length: Incomplete
    def __init__(self, root: Union[str, Sequence[str], IterDataPipe] = ..., masks: Union[str, List[str]] = ..., *, recursive: bool = ..., abspath: bool = ..., non_deterministic: bool = ..., length: int = ...) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self): ...
