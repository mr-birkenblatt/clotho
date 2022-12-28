# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable, Iterator, Optional, Tuple, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


T_co = TypeVar('T_co', covariant=True)


class ConcaterIterDataPipe(IterDataPipe):
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]
    def __init__(self, *datapipes: IterDataPipe) -> None: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...


class ForkerIterDataPipe(IterDataPipe):

    def __new__(
        cls, datapipe: IterDataPipe, num_instances: int,
        buffer_size: int = ...): ...


class _ForkerIterDataPipe(IterDataPipe):
    main_datapipe: Incomplete
    num_instances: Incomplete
    buffer: Incomplete
    buffer_size: Incomplete
    child_pointers: Incomplete
    slowest_ptr: int
    leading_ptr: int
    end_ptr: Incomplete

    def __init__(
        self, datapipe: IterDataPipe, num_instances: int,
        buffer_size: int = ...) -> None: ...

    def __len__(self): ...
    def get_next_element_by_instance(self, instance_id: int): ...
    def is_every_instance_exhausted(self) -> bool: ...
    def reset(self) -> None: ...
    def __del__(self) -> None: ...


class _ChildDataPipe(IterDataPipe):
    main_datapipe: Incomplete
    instance_id: Incomplete

    def __init__(
        self, main_datapipe: IterDataPipe, instance_id: int) -> None: ...

    def __iter__(self): ...
    def __len__(self): ...


class DemultiplexerIterDataPipe(IterDataPipe):

    def __new__(
        cls, datapipe: IterDataPipe, num_instances: int,
        classifier_fn: Callable[[T_co], Optional[int]],
        drop_none: bool = ..., buffer_size: int = ...): ...


class _DemultiplexerIterDataPipe(IterDataPipe):
    main_datapipe: Incomplete
    num_instances: Incomplete
    buffer_size: Incomplete
    current_buffer_usage: int
    child_buffers: Incomplete
    classifier_fn: Incomplete
    drop_none: Incomplete
    main_datapipe_exhausted: bool

    def __init__(
        self, datapipe: IterDataPipe[T_co], num_instances: int,
        classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool,
        buffer_size: int) -> None: ...

    def get_next_element_by_instance(self, instance_id: int): ...
    def is_every_instance_exhausted(self) -> bool: ...
    def reset(self) -> None: ...
    def __del__(self) -> None: ...


class MultiplexerIterDataPipe(IterDataPipe):
    datapipes: Incomplete
    length: Incomplete
    buffer: Incomplete
    def __init__(self, *datapipes) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...
    def reset(self) -> None: ...
    def __del__(self) -> None: ...


class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]
    def __init__(self, *datapipes: IterDataPipe) -> None: ...
    def __iter__(self) -> Iterator[Tuple[T_co]]: ...
    def __len__(self) -> int: ...
