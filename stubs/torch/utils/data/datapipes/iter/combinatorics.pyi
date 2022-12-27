# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from typing import Dict, Iterator, Optional, Tuple, Type, TypeVar

from _typeshed import Incomplete
from torch.utils.data import Sampler
from torch.utils.data.datapipes.datapipe import IterDataPipe


T_co = TypeVar('T_co', covariant=True)


class SamplerIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    sampler: Sampler
    sampler_args: Incomplete
    sampler_kwargs: Incomplete

    def __init__(
        self, datapipe: IterDataPipe, sampler: Type[Sampler] = ...,
        sampler_args: Optional[Tuple] = ...,
        sampler_kwargs: Optional[Dict] = ...) -> None: ...

    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


class ShufflerIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe[T_co]
    buffer_size: int

    def __init__(
        self, datapipe: IterDataPipe[T_co], *, buffer_size: int = ...,
        unbatch_level: int = ...) -> None: ...

    def set_shuffle(self, shuffle: bool = ...): ...
    def set_seed(self, seed: int): ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...
    def reset(self) -> None: ...
    def __del__(self) -> None: ...
