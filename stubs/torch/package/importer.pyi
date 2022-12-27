# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import abc
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from _typeshed import Incomplete

from ._mangling import demangle as demangle
from ._mangling import get_mangle_prefix as get_mangle_prefix
from ._mangling import is_mangled as is_mangled


class ObjNotFoundError(Exception):
    ...


class ObjMismatchError(Exception):
    ...


class Importer(ABC, metaclass=abc.ABCMeta):
    modules: Dict[str, ModuleType]
    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType: ...

    def get_name(
        self, obj: Any, name: Optional[str] = ...) -> Tuple[str, str]: ...

    def whichmodule(self, obj: Any, name: str) -> str: ...


class _SysImporter(Importer):
    def import_module(self, module_name: str): ...
    def whichmodule(self, obj: Any, name: str) -> str: ...


sys_importer: Incomplete


class OrderedImporter(Importer):
    def __init__(self, *args) -> None: ...
    def import_module(self, module_name: str) -> ModuleType: ...
    def whichmodule(self, obj: Any, name: str) -> str: ...
