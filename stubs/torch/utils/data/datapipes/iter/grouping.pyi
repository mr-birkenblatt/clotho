# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Callable, Iterator, Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe


T_co = TypeVar('T_co', covariant=True)


class ShardingFilterIterDataPipe(IterDataPipe):
    source_datapipe: Incomplete
    num_of_instances: int
    instance_id: int
    def __init__(self, source_datapipe: IterDataPipe) -> None: ...
    def is_shardable(self): ...
    def apply_sharding(self, num_of_instances, instance_id) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...


class BatcherIterDataPipe(IterDataPipe[DataChunk]):
    datapipe: IterDataPipe
    batch_size: int
    drop_last: bool
    length: Optional[int]
    wrapper_class: Incomplete

    def __init__(
        self, datapipe: IterDataPipe, batch_size: int,
        drop_last: bool = ..., wrapper_class=...) -> None: ...

    def __iter__(self) -> Iterator[DataChunk]: ...
    def __len__(self) -> int: ...


class UnBatcherIterDataPipe(IterDataPipe):
    datapipe: Incomplete
    unbatch_level: Incomplete

    def __init__(
        self, datapipe: IterDataPipe, unbatch_level: int = ...) -> None: ...

    def __iter__(self): ...


class GrouperIterDataPipe(IterDataPipe[DataChunk]):
    datapipe: Incomplete
    group_key_fn: Incomplete
    max_buffer_size: Incomplete
    buffer_elements: Incomplete
    curr_buffer_size: int
    group_size: Incomplete
    guaranteed_group_size: Incomplete
    drop_remaining: Incomplete
    wrapper_class: Incomplete

    def __init__(
        self, datapipe: IterDataPipe[T_co], group_key_fn: Callable, *,
        buffer_size: int = ..., group_size: Optional[int] = ...,
        guaranteed_group_size: Optional[int] = ...,
        drop_remaining: bool = ...) -> None: ...

    def __iter__(self): ...
    def reset(self) -> None: ...
    def __del__(self) -> None: ...
