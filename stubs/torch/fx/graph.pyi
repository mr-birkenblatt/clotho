# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type

import torch.utils._pytree as pytree
from _typeshed import Incomplete

from ._compatibility import compatibility as compatibility
from ._symbolic_trace import Tracer as Tracer
from .graph_module import GraphModule as GraphModule
from .node import Argument as Argument
from .node import map_arg as map_arg
from .node import Node as Node
from .node import Target as Target


TransformCodeFunc = Callable[[List[str]], List[str]]


class _CustomBuiltin(NamedTuple):
    import_str: str
    obj: Any


class _Namespace:
    def __init__(self) -> None: ...
    def create_name(self, candidate: str, obj: Optional[Any]) -> str: ...
    def associate_name_with_obj(self, name: str, obj: Any): ...


class PythonCode:
    src: str
    globals: Dict[str, Any]
    def __init__(self, src, globals) -> None: ...


class _InsertPoint:
    graph: Incomplete
    def __init__(self, graph, new_insert) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type, value, tb) -> None: ...


class _node_list:
    graph: Incomplete
    direction: Incomplete
    def __init__(self, graph: Graph, direction: str = ...) -> None: ...
    def __len__(self): ...
    def __iter__(self): ...
    def __reversed__(self): ...


class _PyTreeInfo(NamedTuple):
    orig_args: List[str]
    in_spec: pytree.TreeSpec
    out_spec: Optional[pytree.TreeSpec]


class CodeGen:
    def __init__(self) -> None: ...

    def gen_fn_def(
        self, free_vars: List[str], maybe_return_annotation: str) -> str: ...

    def generate_output(self, output_args: Argument) -> str: ...
    def process_inputs(self, *args: Any) -> Any: ...
    def process_outputs(self, outputs: Any) -> Any: ...
    def additional_globals(self) -> List[Tuple[str, Any]]: ...


class _PyTreeCodeGen(CodeGen):
    pytree_info: Incomplete
    def __init__(self, pytree_info: _PyTreeInfo) -> None: ...
    def process_inputs(self, *inputs: Any) -> Any: ...
    def process_outputs(self, out: Any) -> Any: ...
    def gen_fn_def(self, free_vars, maybe_return_annotation): ...
    def generate_output(self, output_args): ...


class Graph:

    def __init__(
        self, owning_module: Optional['GraphModule'] = ...,
        tracer_cls: Optional[Type['Tracer']] = ...,
        tracer_extras: Optional[Dict[str, Any]] = ...) -> None: ...

    @property
    def owning_module(self): ...
    @owning_module.setter
    def owning_module(self, mod: Optional['GraphModule']): ...
    @property
    def nodes(self) -> _node_list: ...

    def graph_copy(
        self, g: Graph, val_map: Dict[Node, Node],
        return_output_node: bool = ...) -> Optional[Argument]: ...

    def __deepcopy__(self, memo: Incomplete | None = ...) -> Graph: ...

    def create_node(
        self, op: str, target: Target, args: Optional[Tuple['Argument',
                        ...]] = ..., kwargs: Optional[Dict[str,
                        'Argument']] = ..., name: Optional[str] = ...,
        type_expr: Optional[Any] = ...) -> Node: ...

    def process_inputs(self, *args): ...
    def process_outputs(self, out): ...
    def erase_node(self, to_erase: Node) -> None: ...
    def inserting_before(self, n: Optional[Node] = ...): ...
    def inserting_after(self, n: Optional[Node] = ...): ...

    def placeholder(
        self, name: str, type_expr: Optional[Any] = ...,
        default_value: Any = ...) -> Node: ...

    def get_attr(
        self, qualified_name: str, type_expr: Optional[Any] = ...) -> Node: ...

    def call_module(
        self, module_name: str, args: Optional[Tuple['Argument', ...]] = ...,
        kwargs: Optional[Dict[str, 'Argument']] = ...,
        type_expr: Optional[Any] = ...) -> Node: ...

    def call_method(
        self, method_name: str, args: Optional[Tuple['Argument', ...]] = ...,
        kwargs: Optional[Dict[str, 'Argument']] = ...,
        type_expr: Optional[Any] = ...) -> Node: ...

    def call_function(
        self, the_function: Callable[..., Any],
        args: Optional[Tuple['Argument', ...]] = ...,
        kwargs: Optional[Dict[str, 'Argument']] = ...,
        type_expr: Optional[Any] = ...) -> Node: ...

    def node_copy(
        self, node: Node, arg_transform: Callable[[Node],
                'Argument'] = ...) -> Node: ...

    def output(self, result: Argument, type_expr: Optional[Any] = ...): ...
    def python_code(self, root_module: str) -> PythonCode: ...
    def print_tabular(self) -> None: ...
    def lint(self): ...
    def eliminate_dead_code(self): ...
    def set_codegen(self, codegen: CodeGen): ...

    def on_generate_code(
        self, make_transformer: Callable[[Optional[TransformCodeFunc]],
                TransformCodeFunc]): ...


reflectable_magic_methods: Incomplete
magic_methods: Incomplete
inplace_methods: Incomplete
