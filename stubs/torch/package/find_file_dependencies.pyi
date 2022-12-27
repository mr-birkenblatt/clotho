# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import ast
from typing import List, Optional, Tuple

from _typeshed import Incomplete


class _ExtractModuleReferences(ast.NodeVisitor):

    @classmethod
    def run(
        cls, src: str, package: str) -> List[Tuple[str, Optional[str]]]: ...

    package: Incomplete
    references: Incomplete
    def __init__(self, package) -> None: ...
    def visit_Import(self, node) -> None: ...
    def visit_ImportFrom(self, node) -> None: ...
    def visit_Call(self, node) -> None: ...


find_files_source_depends_on: Incomplete
