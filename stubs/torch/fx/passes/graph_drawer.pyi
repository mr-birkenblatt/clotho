# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Dict

import pydot
import torch.fx
from _typeshed import Incomplete
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.passes.shape_prop import TensorMetadata as TensorMetadata


HAS_PYDOT: bool


class FxGraphDrawer:

    def __init__(
        self, graph_module: torch.fx.GraphModule, name: str,
        ignore_getattr: bool = ...,
        skip_node_names_in_args: bool = ...) -> None: ...

    def get_dot_graph(
        self, submod_name: Incomplete | None = ...) -> pydot.Dot: ...

    def get_main_dot_graph(self) -> pydot.Dot: ...
    def get_submod_dot_graph(self, submod_name) -> pydot.Dot: ...
    def get_all_dot_graphs(self) -> Dict[str, pydot.Dot]: ...
