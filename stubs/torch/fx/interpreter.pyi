# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Dict, Optional, Tuple

from _typeshed import Incomplete

from ._compatibility import compatibility as compatibility
from ._symbolic_trace import Tracer as Tracer
from .graph import Graph as Graph
from .graph_module import GraphModule as GraphModule
from .node import Argument as Argument
from .node import map_aggregate as map_aggregate
from .node import map_arg as map_arg
from .node import Node as Node
from .node import Target as Target
from .proxy import Proxy as Proxy


class Interpreter:
    module: Incomplete
    submodules: Incomplete
    env: Incomplete
    garbage_collect_values: Incomplete
    user_to_last_uses: Incomplete

    def __init__(
        self, module: GraphModule, garbage_collect_values: bool = ...): ...

    args_iter: Incomplete

    def run(
        self, *args, initial_env: Optional[Dict[Node, Any]] = ...,
        enable_io_processing: bool = ...) -> Any: ...

    def run_node(self, n: Node) -> Any: ...

    def placeholder(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def get_attr(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def call_method(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def call_module(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def output(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def fetch_attr(self, target: str): ...
    def fetch_args_kwargs_from_env(self, n: Node) -> Tuple[Tuple, Dict]: ...
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument: ...


class Transformer(Interpreter):
    new_graph: Incomplete
    graph: Incomplete
    tracer: Incomplete
    def __init__(self, module): ...

    def placeholder(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Proxy: ...

    def get_attr(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Proxy: ...

    def call_module(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
            Any]) -> Any: ...

    def transform(self) -> GraphModule: ...
