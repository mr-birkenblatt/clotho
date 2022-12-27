# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from io import IOBase
from typing import Iterable, Optional, Tuple

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class FileOpenerIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    datapipe: Incomplete
    mode: Incomplete
    encoding: Incomplete
    length: Incomplete

    def __init__(
        self, datapipe: Iterable[str], mode: str = ...,
        encoding: Optional[str] = ..., length: int = ...) -> None: ...

    def __iter__(self): ...
    def __len__(self): ...


class FileLoaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):

    def __new__(
        cls, datapipe: Iterable[str], mode: str = ..., length: int = ...): ...
