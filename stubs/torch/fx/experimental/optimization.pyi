# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Dict, Iterable, List, Optional, Type

import torch
import torch.fx as fx
import torch.nn as nn
from _typeshed import Incomplete
from torch.fx.node import Argument as Argument
from torch.fx.node import Target as Target
from torch.fx.passes.shape_prop import ShapeProp as ShapeProp
from torch.nn.utils.fusion import fuse_conv_bn_eval as fuse_conv_bn_eval


def matches_module_pattern(
    pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]): ...


def replace_node_module(
    node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module): ...


def fuse(model: torch.nn.Module, inplace: bool = ...) -> torch.nn.Module: ...


def remove_dropout(model: nn.Module) -> nn.Module: ...


def extract_subgraph(
    orig_module: nn.Module, nodes: List[fx.Node], inputs: List[fx.Node],
    outputs: List[fx.Node]): ...


mkldnn_supported: Incomplete
mkldnn_supported_unknown: Incomplete
mkldnn_map: Incomplete


def modules_to_mkldnn(nodes: List[fx.Node], modules: Dict[str, nn.Module]): ...


def reset_modules(
    nodes: List[fx.Node], modules: Dict[str, nn.Module],
    old_modules: Dict[nn.Module, nn.Module]): ...


class MklSubgraph:
    fx_graph: Incomplete
    nodes: Incomplete
    start_nodes: Incomplete
    end_nodes: Incomplete
    def __init__(self, fx_graph: fx.Graph) -> None: ...


def gen_mkl_autotuner(example_inputs, iters: int = ..., warmup: int = ...): ...


def use_mkl_length(graph: MklSubgraph) -> bool: ...


class UnionFind:
    parent: Incomplete
    size: Incomplete
    def __init__(self, n) -> None: ...
    def make_set(self, v: int): ...
    def find(self, v: int) -> int: ...
    def join(self, a: int, b: int): ...


def optimize_for_inference(
    model: torch.nn.Module, pass_config: Optional[Dict[str, Any]] = ...,
    tracer: Type[fx.Tracer] = ...) -> torch.nn.Module: ...
