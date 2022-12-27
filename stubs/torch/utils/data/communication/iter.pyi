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
from torch.utils.data import IterDataPipe


def default_not_available_hook() -> None: ...


class NotAvailable(Exception):
    ...


class InvalidStateResetRequired(Exception):
    ...


class NonBlocking(IterDataPipe):
    not_available_hook: Incomplete
    def __iter__(self): ...
    def __next__(self): ...
    def nonblocking_next(self) -> None: ...
    def reset_iterator(self) -> None: ...
    @staticmethod
    def register_not_available_hook(hook_function) -> None: ...


def EnsureNonBlockingDataPipe(validated_datapipe): ...


def DataPipeBehindQueues(
    source_datapipe, protocol, full_stop: bool = ...,
    blocking_request_get: bool = ...) -> Generator[Incomplete, None, None]: ...


class QueueWrapper(NonBlocking):
    protocol: Incomplete
    counter: int
    def __init__(self, protocol, response_wait_time: float = ...) -> None: ...
    def reset_iterator(self) -> None: ...
    def nonblocking_next(self): ...
