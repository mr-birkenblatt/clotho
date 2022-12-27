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
from torch._utils import ExceptionWrapper as ExceptionWrapper

from . import HAS_NUMPY as HAS_NUMPY
from . import IS_WINDOWS as IS_WINDOWS
from . import MP_STATUS_CHECK_INTERVAL as MP_STATUS_CHECK_INTERVAL
from . import signal_handling as signal_handling


class ManagerWatchdog:
    manager_pid: Incomplete
    kernel32: Incomplete
    manager_handle: Incomplete
    manager_dead: bool
    def __init__(self) -> None: ...
    def is_alive(self): ...


class ManagerWatchdog:
    manager_pid: Incomplete
    manager_dead: bool
    def __init__(self) -> None: ...
    def is_alive(self): ...


class WorkerInfo:
    def __init__(self, **kwargs) -> None: ...
    def __setattr__(self, key, val): ...


def get_worker_info(): ...


class _IterableDatasetStopIteration:
    worker_id: int
    def __init__(self, worker_id) -> None: ...


class _ResumeIteration:
    seed: Optional[int]
    def __init__(self, seed) -> None: ...
