import abc
from typing import Any, Callable, Dict, Protocol


class TimerClass(Protocol):
    def __init__(self, stmt: str, setup: str, timer: Callable[[], float], globals: Dict[str, Any], **kwargs: Any) -> None: ...
    def timeit(self, number: int) -> float: ...

class TimeitModuleType(Protocol):
    def timeit(self, number: int) -> float: ...

class CallgrindModuleType(metaclass=abc.ABCMeta):
    __file__: str
    __name__: str
