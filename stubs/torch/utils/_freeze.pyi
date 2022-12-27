# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import types
from pathlib import Path
from typing import List

from _typeshed import Incomplete


PATH_MARKER: str
MAIN_INCLUDES: str
MAIN_PREFIX_TEMPLATE: str
FAKE_PREFIX: Incomplete
MAIN_SUFFIX: str
DENY_LIST: Incomplete
NUM_BYTECODE_FILES: int


def indent_msg(fn): ...


class FrozenModule:
    module_name: str
    c_name: str
    size: int
    bytecode: bytes
    def __init__(self, module_name, c_name, size, bytecode) -> None: ...


class Freezer:
    frozen_modules: Incomplete
    indent: int
    verbose: Incomplete
    def __init__(self, verbose: bool) -> None: ...
    def msg(self, path: Path, code: str): ...
    def write_bytecode(self, install_root) -> None: ...
    def write_main(self, install_root, oss, symbol_name) -> None: ...
    def write_frozen(self, m: FrozenModule, outfp): ...
    def compile_path(self, path: Path, top_package_path: Path): ...
    def compile_package(self, path: Path, top_package_path: Path): ...

    def get_module_qualname(
        self, file_path: Path, top_package_path: Path) -> List[str]: ...

    def compile_string(self, file_content: str) -> types.CodeType: ...
    def compile_file(self, path: Path, top_package_path: Path): ...
