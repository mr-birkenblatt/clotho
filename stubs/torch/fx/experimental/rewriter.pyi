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
    def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = ...) -> Graph: ...
