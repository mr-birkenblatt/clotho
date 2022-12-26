# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Callable, Iterator, Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


T_co = TypeVar('T_co', covariant=True)


class FilterIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    filter_fn: Callable
    drop_empty_batches: bool
    input_col: Incomplete

    def __init__(
        self,
        datapipe: IterDataPipe,
        filter_fn: Callable,
        drop_empty_batches: Optional[bool] = ...,
        input_col: Incomplete | None = ...) -> None: ...

    def __iter__(self) -> Iterator[T_co]: ...
