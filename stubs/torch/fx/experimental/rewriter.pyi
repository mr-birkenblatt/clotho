# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import ast
from types import FunctionType
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch._sources import normalize_source_lines as normalize_source_lines
from torch.fx._symbolic_trace import Tracer as Tracer
from torch.fx.graph import Graph as Graph


class AST_Rewriter(ast.NodeTransformer):
    def rewrite(self, fn: FunctionType): ...
    def visit_Assert(self, node): ...
    def visit_AnnAssign(self, node): ...


class RewritingTracer(Tracer):

    def trace(
        self, root: Union[torch.nn.Module, Callable],
        concrete_args: Optional[Dict[str, Any]] = ...) -> Graph: ...
