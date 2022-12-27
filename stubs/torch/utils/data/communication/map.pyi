# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from collections.abc import Generator

from _typeshed import Incomplete
from torch.utils.data import MapDataPipe


def default_not_available_hook() -> None: ...


class NotAvailable(Exception):
    ...


class NonBlockingMap(MapDataPipe):
    not_available_hook: Incomplete
    def __getitem__(self, index): ...
    def __len__(self): ...
    def nonblocking_len(self) -> None: ...
    def nonblocking_getitem(self, index) -> None: ...
    @staticmethod
    def register_not_available_hook(hook_function) -> None: ...


def EnsureNonBlockingMapDataPipe(validated_datapipe): ...


def DataPipeBehindQueues(
    source_datapipe, protocol, full_stop: bool = ...,
    blocking_request_get: bool = ...) -> Generator[Incomplete, None, None]: ...


class QueueWrapperForMap(NonBlockingMap):
    protocol: Incomplete
    counter: int
    def __init__(self, protocol, response_wait_time: float = ...) -> None: ...
    def nonblocking_getitem(self, index): ...
    def nonblocking_len(self): ...
