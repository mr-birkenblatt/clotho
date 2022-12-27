# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Dict, Set, Union

import torch
from _typeshed import Incomplete
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Graph as Graph


class FusedGraphModule(GraphModule):
    preserved_attr_names: Incomplete

    def __init__(
        self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph,
        preserved_attr_names: Set[str]) -> None: ...

    def __deepcopy__(self, memo): ...


class ObservedGraphModule(GraphModule):
    preserved_attr_names: Incomplete

    def __init__(
        self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph,
        preserved_attr_names: Set[str]) -> None: ...

    def __deepcopy__(self, memo): ...


def is_observed_module(module: Any) -> bool: ...


class ObservedStandaloneGraphModule(ObservedGraphModule):

    def __init__(
        self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph,
        preserved_attr_names: Set[str]) -> None: ...

    def __deepcopy__(self, memo): ...


def is_observed_standalone_module(module: Any) -> bool: ...


class QuantizedGraphModule(GraphModule):
    preserved_attr_names: Incomplete

    def __init__(
        self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph,
        preserved_attr_names: Set[str]) -> None: ...

    def __deepcopy__(self, memo): ...
