# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from collections.abc import Generator

from _typeshed import Incomplete

from . import check_error as check_error
from . import cudart as cudart


DEFAULT_FLAGS: Incomplete


def init(
    output_file, flags: Incomplete | None = ...,
        output_mode: str = ...) -> None: ...


def start() -> None: ...


def stop() -> None: ...


def profile() -> Generator[None, None, None]: ...
