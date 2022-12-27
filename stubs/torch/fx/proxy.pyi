# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from _typeshed import Incomplete

from ._compatibility import compatibility as compatibility
from .graph import Graph as Graph
from .graph import magic_methods as magic_methods
from .graph import reflectable_magic_methods as reflectable_magic_methods
from .node import Argument as Argument
from .node import base_types as base_types
from .node import map_aggregate as map_aggregate
from .node import Node as Node
from .node import Target as Target
from .operator_schemas import (
    check_for_mutable_operation as check_for_mutable_operation,
)


class TracerBase:
    graph: Graph
    record_stack_traces: bool
    check_mutable_operations: bool
    trace_asserts: bool
    proxy_buffer_attributes: bool
    traced_func_name: str

    def create_node(
        self, kind: str, target: Target, args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument], name: Optional[str] = ...,
        type_expr: Optional[Any] = ...) -> Node: ...

    def proxy(self, node: Node) -> Proxy: ...

    def create_proxy(
        self, kind: str, target: Target, args: Tuple[Any, ...],
        kwargs: Dict[str, Any], name: Optional[str] = ...,
        type_expr: Optional[Any] = ..., proxy_factory_fn: Callable[[Node],
            'Proxy'] = ...): ...

    def create_arg(self, a: Any) -> Argument: ...
    def to_bool(self, obj: Proxy) -> bool: ...
    def iter(self, obj: Proxy) -> Iterator: ...
    def keys(self, obj: Proxy) -> Any: ...


class GraphAppendingTracer(TracerBase):
    graph: Incomplete
    def __init__(self, graph: Graph) -> None: ...


def assert_fn(x) -> None: ...


class TraceError(ValueError):
    ...


class Proxy:
    tracer: Incomplete
    node: Incomplete

    def __init__(
        self, node: Node, tracer: Optional[TracerBase] = ...) -> None: ...

    def __getattr__(self, k) -> Attribute: ...
    def __call__(self, *args, **kwargs) -> Proxy: ...
    def __iter__(self) -> Iterable['Proxy']: ...
    def __bool__(self) -> bool: ...
    def keys(self): ...
    def __len__(self) -> None: ...

    @classmethod
    def __torch_function__(
        cls, orig_method, types, args: Incomplete | None = ...,
        kwargs: Incomplete | None = ...): ...


class Attribute(Proxy):
    root: Incomplete
    attr: Incomplete
    tracer: Incomplete
    def __init__(self, root: Proxy, attr: str) -> None: ...
    @property
    def node(self): ...
    def __call__(self, *args, **kwargs): ...


class ParameterProxy(Proxy):
    param: Incomplete
    name: Incomplete

    def __init__(
        self, tracer: TracerBase, node: Node, name, param) -> None: ...

    @property
    def shape(self): ...
    def size(self): ...
    def dim(self): ...
    @property
    def ndim(self): ...
    def numel(self): ...
    def nelement(self): ...
