# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Iterable, Optional, Sequence, TypeVar, Union

from _typeshed import Incomplete

from . import Dataset, Sampler


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
default_collate: _collate_fn_t
default_convert: Incomplete
get_worker_info: Incomplete


class _DatasetKind:
    Map: int
    Iterable: int

    @staticmethod
    def create_fetcher(
        kind, dataset, auto_collation, collate_fn, drop_last): ...


class _InfiniteConstantSampler(Sampler):
    def __init__(self) -> None: ...
    def __iter__(self): ...


class DataLoader:
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: int
    worker_init_fn: Incomplete
    batch_sampler: Incomplete
    generator: Incomplete
    collate_fn: Incomplete
    persistent_workers: Incomplete

    def __init__(
        self, dataset: Dataset[T_co], batch_size: Optional[int] = ...,
        shuffle: Optional[bool] = ..., sampler: Union[Sampler, Iterable,
                None] = ..., batch_sampler: Union[Sampler[Sequence],
                Iterable[Sequence], None] = ..., num_workers: int = ...,
        collate_fn: Optional[_collate_fn_t] = ..., pin_memory: bool = ...,
        drop_last: bool = ..., timeout: float = ...,
        worker_init_fn: Optional[_worker_init_fn_t] = ...,
        multiprocessing_context: Incomplete | None = ...,
        generator: Incomplete | None = ..., *, prefetch_factor: int = ...,
        persistent_workers: bool = ...,
        pin_memory_device: str = ...) -> None: ...

    @property
    def multiprocessing_context(self): ...
    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context) -> None: ...
    def __setattr__(self, attr, val) -> None: ...
    def __iter__(self) -> _BaseDataLoaderIter: ...
    def __len__(self) -> int: ...
    def check_worker_number_rationality(self): ...


class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None: ...
    def __iter__(self) -> _BaseDataLoaderIter: ...
    def __next__(self) -> Any: ...
    next: Incomplete
    def __len__(self) -> int: ...


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None: ...


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader) -> None: ...
    def __del__(self) -> None: ...
