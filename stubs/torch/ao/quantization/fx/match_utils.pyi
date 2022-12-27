# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node

from ..qconfig import QConfigAny as QConfigAny
from ..utils import MatchAllNode as MatchAllNode
from .graph_module import (
    is_observed_standalone_module as is_observed_standalone_module,
)
from .quantization_patterns import QuantizeHandler as QuantizeHandler


MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
        QConfigAny]


def is_match(modules, node, pattern, max_uses=...): ...


def find_matches(
    graph: Graph, modules: Dict[str, torch.nn.Module],
        patterns: Dict[Pattern, QuantizeHandler],
        root_node_getter_mapping: Dict[Pattern, Callable],
        qconfig_map: Dict[str, QConfigAny],
        standalone_module_names: List[str] = ...,
        standalone_module_classes: List[Callable] = ...,
        custom_module_classes: List[Any] = ...) -> Dict[str, MatchResult]: ...
