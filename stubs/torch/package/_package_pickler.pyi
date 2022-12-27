# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from pickle import _Pickler

from _typeshed import Incomplete

from .importer import Importer as Importer
from .importer import ObjMismatchError as ObjMismatchError
from .importer import ObjNotFoundError as ObjNotFoundError
from .importer import sys_importer as sys_importer


class PackagePickler(_Pickler):
    importer: Incomplete
    dispatch: Incomplete
    def __init__(self, importer: Importer, *args, **kwargs) -> None: ...
    def save_global(self, obj, name: Incomplete | None = ...) -> None: ...


def create_pickler(data_buf, importer, protocol: int = ...): ...
