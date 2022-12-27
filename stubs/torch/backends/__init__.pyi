# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import types

from _typeshed import Incomplete


def disable_global_flags() -> None: ...


def flags_frozen(): ...


class ContextProp:
    getter: Incomplete
    setter: Incomplete
    def __init__(self, getter, setter) -> None: ...
    def __get__(self, obj, objtype): ...
    def __set__(self, obj, val) -> None: ...


class PropModule(types.ModuleType):
    m: Incomplete
    def __init__(self, m, name) -> None: ...
    def __getattr__(self, attr): ...
