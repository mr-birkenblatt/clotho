# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from collections.abc import Generator

from .dispatch import dispatch as dispatch
from .utils import hashable as hashable


class Var:
    def __new__(cls, *token): ...
    def __eq__(self, other): ...
    def __hash__(self): ...


def var(): ...


def vars(): ...


def isvar(v): ...


def variables(*variables) -> Generator[None, None, None]: ...
