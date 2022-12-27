# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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
