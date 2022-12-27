# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from threading import Event
from typing import Dict, TextIO

from _typeshed import Incomplete


log: Incomplete


def tail_logfile(
    header: str, file: str, dst: TextIO, finished: Event,
    interval_sec: float): ...


class TailLog:

    def __init__(
        self, name: str, log_files: Dict[int, str], dst: TextIO,
        interval_sec: float = ...) -> None: ...

    def start(self) -> TailLog: ...
    def stop(self) -> None: ...
    def stopped(self) -> bool: ...
