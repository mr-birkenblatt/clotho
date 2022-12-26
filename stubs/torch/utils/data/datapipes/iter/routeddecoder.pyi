# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from io import BufferedIOBase
from typing import Any, Callable, Iterable, Iterator, Tuple

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class RoutedDecoderIterDataPipe(IterDataPipe[Tuple[str, Any]]):
    datapipe: Incomplete
    decoder: Incomplete

    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        *handlers: Callable, key_fn: Callable = ...) -> None: ...

    def add_handler(self, *handler: Callable) -> None: ...
    def __iter__(self) -> Iterator[Tuple[str, Any]]: ...
    def __len__(self) -> int: ...
