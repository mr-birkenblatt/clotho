# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

from _typeshed import Incomplete
from torch import Tensor


T_co = TypeVar('T_co', covariant=True)


class Sampler:
    def __init__(self, data_source: Optional[Sized]) -> None: ...
    def __iter__(self) -> Iterator[T_co]: ...


class SequentialSampler(Sampler[int]):
    data_source: Sized
    def __init__(self, data_source: Sized) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...


class RandomSampler(Sampler[int]):
    data_source: Sized
    replacement: bool
    generator: Incomplete

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = ...,
        num_samples: Optional[int] = ...,
        generator: Incomplete | None = ...) -> None: ...

    @property
    def num_samples(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...


class SubsetRandomSampler(Sampler[int]):
    indices: Sequence[int]
    generator: Incomplete

    def __init__(
        self,
        indices: Sequence[int],
        generator: Incomplete | None = ...) -> None: ...

    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...


class WeightedRandomSampler(Sampler[int]):
    weights: Tensor
    num_samples: int
    replacement: bool
    generator: Incomplete

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = ...,
        generator: Incomplete | None = ...) -> None: ...

    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...


class BatchSampler(Sampler[List[int]]):
    sampler: Incomplete
    batch_size: Incomplete
    drop_last: Incomplete

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool) -> None: ...

    def __iter__(self) -> Iterator[List[int]]: ...
    def __len__(self) -> int: ...
