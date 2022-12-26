import ast

import torch
from _typeshed import Incomplete


class AttributeTypeIsSupportedChecker(ast.NodeVisitor):
    using_deprecated_ast: Incomplete
    class_level_annotations: Incomplete
    visiting_class_level_ann: bool
    def check(self, nn_module: torch.nn.Module) -> None: ...
    def visit_Assign(self, node) -> None: ...
    def visit_AnnAssign(self, node) -> None: ...
    def visit_Call(self, node) -> None: ...
