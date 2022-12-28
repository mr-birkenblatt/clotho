# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete

from .core import reify as reify
from .core import unify as unify
from .unification_tools import first as first
from .unification_tools import groupby as groupby
from .utils import freeze as freeze
from .variable import isvar as isvar


class Dispatcher:
    name: Incomplete
    funcs: Incomplete
    ordering: Incomplete
    def __init__(self, name) -> None: ...
    def add(self, signature, func) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def resolve(self, args): ...
    def register(self, *signature): ...


class VarDispatcher(Dispatcher):
    def __call__(self, *args, **kwargs): ...


global_namespace: Incomplete


def match(*signature, **kwargs): ...


def supercedes(a, b): ...


def edge(a, b, tie_breaker=...): ...


def ordering(signatures): ...
