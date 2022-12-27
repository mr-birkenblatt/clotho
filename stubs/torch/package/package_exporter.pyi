# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from ._digraph import DiGraph as DiGraph
from ._mangling import demangle as demangle
from ._mangling import is_mangled as is_mangled
from ._package_pickler import create_pickler as create_pickler
from ._stdlib import is_stdlib_module as is_stdlib_module
from .find_file_dependencies import (
    find_files_source_depends_on as find_files_source_depends_on,
)
from .glob_group import GlobGroup as GlobGroup
from .glob_group import GlobPattern as GlobPattern
from .importer import Importer as Importer


        OrderedImporter as OrderedImporter, sys_importer as sys_importer
from enum import Enum
from pathlib import Path

from _typeshed import Incomplete
from torch.serialization import location_tag as location_tag


        normalize_storage_type as normalize_storage_type
from typing import Any, BinaryIO, List, Sequence, Union

from torch.types import Storage as Storage
from torch.utils.hooks import RemovableHandle as RemovableHandle


ActionHook: Incomplete


class _ModuleProviderAction(Enum):
    INTERN: int
    EXTERN: int
    MOCK: int
    DENY: int
    REPACKAGED_MOCK_MODULE: int
    SKIP: int


class PackagingErrorReason(Enum):
    IS_EXTENSION_MODULE: str
    NO_DUNDER_FILE: str
    SOURCE_FILE_NOT_FOUND: str
    DEPENDENCY_RESOLUTION_FAILED: str
    NO_ACTION: str
    DENIED: str
    MOCKED_BUT_STILL_USED: str


class _PatternInfo:
    action: _ModuleProviderAction
    allow_empty: bool
    was_matched: bool
    def __init__(self, action, allow_empty) -> None: ...


class EmptyMatchError(Exception):
    ...


class PackagingError(Exception):
    dependency_graph: Incomplete
    def __init__(self, dependency_graph: DiGraph) -> None: ...


class PackageExporter:
    importer: Importer
    buffer: Incomplete
    zip_file: Incomplete
    serialized_reduces: Incomplete
    dependency_graph: Incomplete
    script_module_serializer: Incomplete
    storage_context: Incomplete
    patterns: Incomplete

    def __init__(
        self, f: Union[str, Path, BinaryIO], importer: Union[Importer,
        Sequence[Importer]] = ...) -> None: ...

    def save_source_file(
        self, module_name: str, file_or_directory: str,
        dependencies: bool = ...): ...

    def get_unique_id(self) -> str: ...

    def save_source_string(
        self, module_name: str, src: str, is_package: bool = ...,
        dependencies: bool = ...): ...

    def add_dependency(self, module_name: str, dependencies: bool = ...): ...
    def save_module(self, module_name: str, dependencies: bool = ...): ...

    def save_pickle(
        self, package: str, resource: str, obj: Any,
        dependencies: bool = ..., pickle_protocol: int = ...): ...

    def save_text(self, package: str, resource: str, text: str): ...
    def save_binary(self, package, resource, binary: bytes): ...
    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle: ...
    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle: ...
    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle: ...

    def intern(
        self, include: GlobPattern, *, exclude: GlobPattern = ...,
        allow_empty: bool = ...): ...

    def mock(
        self, include: GlobPattern, *, exclude: GlobPattern = ...,
        allow_empty: bool = ...): ...

    def extern(
        self, include: GlobPattern, *, exclude: GlobPattern = ...,
        allow_empty: bool = ...): ...

    def deny(self, include: GlobPattern, *, exclude: GlobPattern = ...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def close(self) -> None: ...
    def dependency_graph_string(self) -> str: ...
    def externed_modules(self) -> List[str]: ...
    def interned_modules(self) -> List[str]: ...
    def mocked_modules(self) -> List[str]: ...
    def denied_modules(self) -> List[str]: ...
    def get_rdeps(self, module_name: str) -> List[str]: ...
    def all_paths(self, src: str, dst: str) -> str: ...
