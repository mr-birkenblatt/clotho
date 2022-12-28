# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import abc
import signal
from enum import IntFlag
from multiprocessing import synchronize as synchronize
from typing import Any, Callable, Dict, Optional, Tuple, Union

from _typeshed import Incomplete
from torch.distributed.elastic.multiprocessing.errors import (
    ProcessFailure as ProcessFailure,
)
from torch.distributed.elastic.multiprocessing.errors import record as record
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stderr as redirect_stderr,
)
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stdout as redirect_stdout,
)
from torch.distributed.elastic.multiprocessing.tail_log import (
    TailLog as TailLog,
)


IS_WINDOWS: Incomplete
IS_MACOS: Incomplete
log: Incomplete


class SignalException(Exception):
    sigval: Incomplete
    def __init__(self, msg: str, sigval: signal.Signals) -> None: ...


class Std(IntFlag):
    NONE: int
    OUT: int
    ERR: int
    ALL: Incomplete
    @classmethod
    def from_str(cls, vm: str) -> Union['Std', Dict[int, 'Std']]: ...


def to_map(
    val_or_map: Union[Std, Dict[int, Std]], local_world_size: int) -> Dict[
        int, Std]: ...


class RunProcsResult:
    return_values: Dict[int, Any]
    failures: Dict[int, ProcessFailure]
    stdouts: Dict[int, str]
    stderrs: Dict[int, str]
    def is_failed(self) -> bool: ...
    def __init__(self, return_values, failures, stdouts, stderrs) -> None: ...


class PContext(abc.ABC, metaclass=abc.ABCMeta):
    name: Incomplete
    entrypoint: Incomplete
    args: Incomplete
    envs: Incomplete
    stdouts: Incomplete
    stderrs: Incomplete
    error_files: Incomplete
    nprocs: Incomplete

    def __init__(
        self, name: str, entrypoint: Union[Callable, str], args: Dict[int,
                Tuple], envs: Dict[int, Dict[str, str]], stdouts: Dict[int,
                str], stderrs: Dict[int, str], tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str], error_files: Dict[int, str]) -> None: ...

    def start(self) -> None: ...

    def wait(
        self, timeout: float = ..., period: float = ...) -> Optional[
            RunProcsResult]: ...

    @abc.abstractmethod
    def pids(self) -> Dict[int, int]: ...

    def close(
        self, death_sig: Optional[signal.Signals] = ...,
        timeout: int = ...) -> None: ...


def get_std_cm(std_rd: str, redirect_fn): ...


class MultiprocessContext(PContext):
    start_method: Incomplete

    def __init__(
        self, name: str, entrypoint: Callable, args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]], stdouts: Dict[int, str],
        stderrs: Dict[int, str], tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str], error_files: Dict[int, str],
        start_method: str) -> None: ...

    def pids(self) -> Dict[int, int]: ...


class SubprocessHandler:
    proc: Incomplete

    def __init__(
        self, entrypoint: str, args: Tuple, env: Dict[str, str], stdout: str,
        stderr: str) -> None: ...

    def close(self, death_sig: Optional[signal.Signals] = ...) -> None: ...


class SubprocessContext(PContext):
    subprocess_handlers: Incomplete

    def __init__(
        self, name: str, entrypoint: str, args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]], stdouts: Dict[int, str],
        stderrs: Dict[int, str], tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str], error_files: Dict[int, str]) -> None: ...

    def pids(self) -> Dict[int, int]: ...
