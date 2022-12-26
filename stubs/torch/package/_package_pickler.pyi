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
