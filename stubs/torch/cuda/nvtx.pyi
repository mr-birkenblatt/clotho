# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from collections.abc import Generator

from _typeshed import Incomplete


class _NVTXStub:
    rangePushA: Incomplete
    rangePop: Incomplete
    markA: Incomplete


def range_push(msg): ...


def range_pop(): ...


def range_start(msg) -> int: ...


def range_end(range_id) -> None: ...


def mark(msg): ...


def range(msg, *args, **kwargs) -> Generator[None, None, None]: ...
