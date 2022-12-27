# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

from _typeshed import Incomplete

from ... import Generator, Tensor


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Dataset:
    def __getitem__(self, index) -> T_co: ...
    def __add__(self, other: Dataset[T_co]) -> ConcatDataset[T_co]: ...


class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]: ...
    def __add__(self, other: Dataset[T_co]): ...


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    tensors: Tuple[Tensor, ...]
    def __init__(self, *tensors: Tensor) -> None: ...
    def __getitem__(self, index): ...
    def __len__(self): ...


class ConcatDataset(Dataset[T_co]):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]
    @staticmethod
    def cumsum(sequence): ...
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __len__(self): ...
    def __getitem__(self, idx): ...
    @property
    def cummulative_sizes(self): ...


class ChainDataset(IterableDataset):
    datasets: Incomplete
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...


class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(
        self, dataset: Dataset[T_co], indices: Sequence[int]) -> None: ...

    def __getitem__(self, idx): ...
    def __len__(self): ...


def random_split(
    dataset: Dataset[T], lengths: Sequence[int],
        generator: Optional[Generator] = ...) -> List[Subset[T]]: ...
