# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Iterator, Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data import Dataset, Sampler


T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
    dataset: Incomplete
    num_replicas: Incomplete
    rank: Incomplete
    epoch: int
    drop_last: Incomplete
    num_samples: Incomplete
    total_size: Incomplete
    shuffle: Incomplete
    seed: Incomplete

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = ...,
        rank: Optional[int] = ...,
        shuffle: bool = ...,
        seed: int = ...,
        drop_last: bool = ...) -> None: ...

    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...
    def set_epoch(self, epoch: int) -> None: ...
