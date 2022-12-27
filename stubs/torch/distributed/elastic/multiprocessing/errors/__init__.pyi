# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Dict, Optional, Tuple, TypeVar

from _typeshed import Incomplete
from torch.distributed.elastic.utils.logging import get_logger as get_logger

from .error_handler import ErrorHandler as ErrorHandler
from .handlers import get_error_handler as get_error_handler


log: Incomplete
JSON = Dict
T = TypeVar('T')


class ProcessFailure:
    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: JSON
    message: str
    timestamp: int
    def __post_init__(self) -> None: ...
    def signal_name(self) -> str: ...
    def timestamp_isoformat(self): ...
    def __init__(self, local_rank, pid, exitcode, error_file) -> None: ...


GlobalRank = int


class ChildFailedError(Exception):
    name: Incomplete
    failures: Incomplete

    def __init__(
        self, name: str, failures: Dict[GlobalRank,
            ProcessFailure]) -> None: ...

    def get_first_failure(self) -> Tuple[GlobalRank, ProcessFailure]: ...

    def format_msg(
        self, boarder_delim: str = ..., section_delim: str = ...): ...


def record(
    fn: Callable[..., T],
        error_handler: Optional[ErrorHandler] = ...) -> Callable[..., T]: ...
