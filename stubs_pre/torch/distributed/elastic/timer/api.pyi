# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import abc
from typing import Any, Dict, List, Optional, Set

from _typeshed import Incomplete


class TimerRequest:
    worker_id: Incomplete
    scope_id: Incomplete
    expiration_time: Incomplete

    def __init__(
        self, worker_id: Any, scope_id: str,
        expiration_time: float) -> None: ...

    def __eq__(self, other): ...


class TimerClient(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None: ...
    @abc.abstractmethod
    def release(self, scope_id: str): ...


class RequestQueue(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def size(self) -> int: ...
    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> List[TimerRequest]: ...


class TimerServer(abc.ABC, metaclass=abc.ABCMeta):

    def __init__(
        self, request_queue: RequestQueue, max_interval: float,
        daemon: bool = ...) -> None: ...

    @abc.abstractmethod
    def register_timers(self, timer_requests: List[TimerRequest]) -> None: ...
    @abc.abstractmethod
    def clear_timers(self, worker_ids: Set[Any]) -> None: ...

    @abc.abstractmethod
    def get_expired_timers(
        self, deadline: float) -> Dict[str, List[TimerRequest]]: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...


def configure(timer_client: TimerClient): ...


def expires(
    after: float, scope: Optional[str] = ...,
    client: Optional[TimerClient] = ...): ...
