# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import abc
from typing import Any, Callable, Dict, Protocol


class TimerClass(Protocol):

    def __init__(
        self, stmt: str, setup: str, timer: Callable[[], float],
        globals: Dict[str, Any], **kwargs: Any) -> None: ...

    def timeit(self, number: int) -> float: ...


class TimeitModuleType(Protocol):
    def timeit(self, number: int) -> float: ...


class CallgrindModuleType(metaclass=abc.ABCMeta):
    __file__: str
    __name__: str
