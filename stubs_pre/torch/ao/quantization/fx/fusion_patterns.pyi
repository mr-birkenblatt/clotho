# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import abc
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.ao.quantization.quantization_types import NodePattern as NodePattern
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node

from ..fuser_method_mappings import (
    get_fuser_method_new as get_fuser_method_new,
)
from .match_utils import MatchAllNode as MatchAllNode


class FuseHandler(ABC, metaclass=abc.ABCMeta):
    def __init__(self, node: Node) -> None: ...

    @abstractmethod
    def fuse(
        self, load_arg: Callable, named_modules: Dict[str, torch.nn.Module],
        fused_graph: Graph, root_node: Node, extra_inputs: List[Any],
        matched_node_pattern: NodePattern, fuse_custom_config_dict: Dict[str,
                Any], fuser_method_mapping: Optional[Dict[Pattern,
                        Union[torch.nn.Sequential, Callable]]],
        is_qat: bool) -> Node: ...


class DefaultFuseHandler(FuseHandler):
    def __init__(self, node: Node) -> None: ...

    def fuse(
        self, load_arg: Callable, named_modules: Dict[str, torch.nn.Module],
        fused_graph: Graph, root_node: Node, extra_inputs: List[Any],
        matched_node_pattern: NodePattern, fuse_custom_config_dict: Dict[str,
                Any], fuser_method_mapping: Optional[Dict[Pattern,
                        Union[torch.nn.Sequential, Callable]]],
        is_qat: bool) -> Node: ...
