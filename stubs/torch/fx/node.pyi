# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from _typeshed import Incomplete
from torch.fx.operator_schemas import ArgsKwargsPair as ArgsKwargsPair
from torch.fx.operator_schemas import normalize_function as normalize_function
from torch.fx.operator_schemas import normalize_module as normalize_module

from ._compatibility import compatibility as compatibility
from .graph import Graph as Graph
from .immutable_collections import immutable_dict as immutable_dict
from .immutable_collections import immutable_list as immutable_list


BaseArgumentTypes: Incomplete
base_types: Incomplete
Target = Union[Callable[..., Any], str]
Argument: Incomplete


class Node:
    graph: Incomplete
    name: Incomplete
    op: Incomplete
    target: Incomplete
    users: Incomplete
    type: Incomplete
    meta: Incomplete

    def __init__(
        self, graph: Graph, name: str, op: str, target: Target,
        args: Tuple['Argument', ...], kwargs: Dict[str, 'Argument'],
        return_type: Optional[Any] = ...): ...

    @property
    def next(self) -> Node: ...
    @property
    def prev(self) -> Node: ...
    def prepend(self, x: Node) -> None: ...
    def append(self, x: Node) -> None: ...
    @property
    def args(self) -> Tuple[Argument, ...]: ...
    @args.setter
    def args(self, a: Tuple[Argument, ...]): ...
    @property
    def kwargs(self) -> Dict[str, Argument]: ...
    @kwargs.setter
    def kwargs(self, k: Dict[str, Argument]): ...
    @property
    def all_input_nodes(self) -> List['Node']: ...
    def update_arg(self, idx: int, arg: Argument) -> None: ...
    def update_kwarg(self, key: str, arg: Argument) -> None: ...
    @property
    def stack_trace(self) -> Optional[str]: ...
    @stack_trace.setter
    def stack_trace(self, trace: Optional[str]): ...

    def format_node(
        self, placeholder_names: Optional[List[str]] = ...,
        maybe_return_typename: Optional[List[str]] = ...) -> Optional[str]: ...

    def replace_all_uses_with(
        self, replace_with: Node, delete_user_cb: Callable[[Node],
            bool] = ...) -> List['Node']: ...

    def is_impure(self): ...

    def normalized_arguments(
        self, root: torch.nn.Module, arg_types: Optional[Tuple[Any]] = ...,
        kwarg_types: Optional[Dict[str, Any]] = ...,
        normalize_to_only_use_kwargs:
        bool = ...) -> Optional[ArgsKwargsPair]: ...

    def replace_input_with(self, old_input: Node, new_input: Node): ...


def map_arg(a: Argument, fn: Callable[[Node], Argument]) -> Argument: ...


def map_aggregate(
    a: Argument, fn: Callable[[Argument], Argument]) -> Argument: ...
