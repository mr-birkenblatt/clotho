# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import multiprocessing as mp
from typing import Any, Dict, List, Set

from .api import RequestQueue as RequestQueue
from .api import TimerClient as TimerClient
from .api import TimerRequest as TimerRequest
from .api import TimerServer as TimerServer


class LocalTimerClient(TimerClient):
    def __init__(self, mp_queue) -> None: ...
    def acquire(self, scope_id, expiration_time) -> None: ...
    def release(self, scope_id) -> None: ...


class MultiprocessingRequestQueue(RequestQueue):
    def __init__(self, mp_queue: mp.Queue) -> None: ...
    def size(self) -> int: ...
    def get(self, size, timeout: float) -> List[TimerRequest]: ...


class LocalTimerServer(TimerServer):

    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = ...,
        daemon: bool = ...) -> None: ...

    def register_timers(self, timer_requests: List[TimerRequest]) -> None: ...
    def clear_timers(self, worker_ids: Set[int]) -> None: ...

    def get_expired_timers(
        self, deadline: float) -> Dict[Any, List[TimerRequest]]: ...
