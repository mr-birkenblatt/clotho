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


class _ClassNamespace(types.ModuleType):
    name: Incomplete
    def __init__(self, name) -> None: ...
    def __getattr__(self, attr): ...


class _Classes(types.ModuleType):
    __file__: str
    def __init__(self) -> None: ...
    def __getattr__(self, name): ...
    @property
    def loaded_libraries(self): ...
    def load_library(self, path) -> None: ...


classes: Incomplete
