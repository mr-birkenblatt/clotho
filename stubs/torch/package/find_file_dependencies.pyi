import ast
from typing import List, Optional, Tuple

from _typeshed import Incomplete


class _ExtractModuleReferences(ast.NodeVisitor):
    @classmethod
    def run(cls, src: str, package: str) -> List[Tuple[str, Optional[str]]]: ...
    package: Incomplete
    references: Incomplete
    def __init__(self, package) -> None: ...
    def visit_Import(self, node) -> None: ...
    def visit_ImportFrom(self, node) -> None: ...
    def visit_Call(self, node) -> None: ...

find_files_source_depends_on: Incomplete
