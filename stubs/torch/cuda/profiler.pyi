from collections.abc import Generator

from _typeshed import Incomplete

from . import check_error as check_error
from . import cudart as cudart


DEFAULT_FLAGS: Incomplete

def init(output_file, flags: Incomplete | None = ..., output_mode: str = ...) -> None: ...
def start() -> None: ...
def stop() -> None: ...
def profile() -> Generator[None, None, None]: ...
