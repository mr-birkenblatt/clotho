# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import types
from collections.abc import Generator
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Optional, Union

import torch
from _typeshed import Incomplete

from ._directory_reader import DirectoryReader as DirectoryReader
from ._importlib import _calc___package__ as _calc___package__
from ._mangling import demangle as demangle
from ._mangling import PackageMangler as PackageMangler
from ._package_unpickler import PackageUnpickler as PackageUnpickler
from .file_structure_representation import Directory as Directory
from .glob_group import GlobPattern as GlobPattern
from .importer import Importer as Importer


class PackageImporter(Importer):
    modules: Dict[str, types.ModuleType]
    zip_reader: Incomplete
    filename: str
    root: Incomplete
    extern_modules: Incomplete
    patched_builtins: Incomplete
    storage_context: Incomplete
    last_map_location: Incomplete
    Unpickler: Incomplete

    def __init__(
        self, file_or_buffer: Union[str, torch._C.PyTorchFileReader, Path,
                BinaryIO], module_allowed: Callable[[str], bool] = ...): ...

    def import_module(self, name: str, package: Incomplete | None = ...): ...
    def load_binary(self, package: str, resource: str) -> bytes: ...

    def load_text(
        self, package: str, resource: str, encoding: str = ...,
        errors: str = ...) -> str: ...

    def load_pickle(
        self, package: str, resource: str,
        map_location: Incomplete | None = ...) -> Any: ...

    def id(self): ...

    def file_structure(
        self, *, include: GlobPattern = ...,
        exclude: GlobPattern = ...) -> Directory: ...

    def python_version(self): ...
    def get_source(self, module_name) -> str: ...
    def get_resource_reader(self, fullname): ...

    def __import__(
        self, name, globals: Incomplete | None = ...,
        locals: Incomplete | None = ..., fromlist=..., level: int = ...): ...


class _PathNode:
    ...


class _PackageNode(_PathNode):
    source_file: Incomplete
    children: Incomplete
    def __init__(self, source_file: Optional[str]) -> None: ...


class _ModuleNode(_PathNode):
    source_file: Incomplete
    def __init__(self, source_file: str) -> None: ...


class _ExternNode(_PathNode):
    ...


def patched_getfile(object): ...


class _PackageResourceReader:
    importer: Incomplete
    fullname: Incomplete
    def __init__(self, importer, fullname) -> None: ...
    def open_resource(self, resource): ...
    def resource_path(self, resource): ...
    def is_resource(self, name): ...
    def contents(self) -> Generator[Incomplete, None, None]: ...
