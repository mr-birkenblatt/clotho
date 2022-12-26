# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Callable, Iterator, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


T_co = TypeVar('T_co', covariant=True)


class MapperIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    fn: Callable
    input_col: Incomplete
    output_col: Incomplete

    def __init__(
        self, datapipe: IterDataPipe, fn: Callable,
        input_col: Incomplete | None = ...,
        output_col: Incomplete | None = ...) -> None: ...

    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


class CollatorIterDataPipe(MapperIterDataPipe):
    def __init__(
        self, datapipe: IterDataPipe, collate_fn: Callable = ...) -> None: ...
