# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
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

    def __init__(
        self,
        root: Union[str, Sequence[str], IterDataPipe] = ...,
        masks: Union[str, List[str]] = ..., *, recursive: bool = ...,
        abspath: bool = ..., non_deterministic: bool = ...,
        length: int = ...) -> None: ...

    def __iter__(self) -> Iterator[str]: ...
    def __len__(self): ...
