# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Optional

from _typeshed import Incomplete


class ProcessException(Exception):
    msg: Incomplete
    error_index: Incomplete
    pid: Incomplete
    def __init__(self, msg: str, error_index: int, pid: int) -> None: ...
    def __reduce__(self): ...


class ProcessRaisedException(ProcessException):
    def __init__(self, msg: str, error_index: int, error_pid: int) -> None: ...


class ProcessExitedException(ProcessException):
    exit_code: Incomplete
    signal_name: Incomplete

    def __init__(
        self, msg: str, error_index: int, error_pid: int, exit_code: int,
        signal_name: Optional[str] = ...) -> None: ...

    def __reduce__(self): ...


class ProcessContext:
    error_queues: Incomplete
    processes: Incomplete
    sentinels: Incomplete
    def __init__(self, processes, error_queues) -> None: ...
    def pids(self): ...
    def join(self, timeout: Incomplete | None = ...): ...


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_queues) -> None: ...


def start_processes(
    fn, args=..., nprocs: int = ..., join: bool = ..., daemon: bool = ...,
    start_method: str = ...): ...


def spawn(
    fn, args=..., nprocs: int = ..., join: bool = ..., daemon: bool = ...,
    start_method: str = ...): ...
