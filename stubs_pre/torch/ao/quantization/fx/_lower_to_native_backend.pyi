# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable, Dict, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.fx import map_arg as map_arg
from torch.fx import Node as Node
from torch.fx.graph import Graph as Graph
from torch.nn.quantized.modules.utils import (
    WeightedQuantizedModule as WeightedQuantizedModule,
)

from ..qconfig import QConfigAny as QConfigAny
from ..quantization_mappings import (
    get_quantized_operator as get_quantized_operator,
)
from .graph_module import QuantizedGraphModule as QuantizedGraphModule
from .utils import collect_producer_nodes as collect_producer_nodes
from .utils import (
    create_node_from_old_node_preserve_meta as create_node_from_old_node_preserve_meta,
)
from .utils import (
    get_linear_prepack_op_for_dtype as get_linear_prepack_op_for_dtype,
)
from .utils import (
    get_new_attr_name_with_prefix as get_new_attr_name_with_prefix,
)
from .utils import get_qconv_prepack_op as get_qconv_prepack_op
from .utils import (
    graph_module_from_producer_nodes as graph_module_from_producer_nodes,
)


QOP_TO_ARG_NAMES_TO_SKIP: Incomplete


def is_fixed_qparams_node(node, modules): ...


def is_default_node(node, modules): ...


def is_copy_node(node, modules): ...


def is_general_tensor_shape_node(node, modules): ...


def is_other_node(node, modules): ...


def is_special_pattern_node(node, modules): ...


def is_dequantize_node(node): ...


def is_getattr_tensor_metadata_node(node): ...


def should_skip_lowering(
    op: torch.fx.node.Node, qconfig_map: Dict[str, QConfigAny]): ...


STATIC_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[WeightedQuantizedModule]]
DYNAMIC_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]]
WEIGHT_ONLY_LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[nn.Module]]
SPECIAL_PATTERN_LOWER_MODULE_MAP: Incomplete
STATIC_LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module],
                    Type[WeightedQuantizedModule]]]
DYNAMIC_LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module],
                    Type[nn.Module]]]
STATIC_LOWER_FUNCTIONAL_MAP: Dict[Callable, Tuple[Callable, Callable]]
WEIGHT_PREPACK_OPS: Set[Callable]
DYNAMIC_LOWER_FUNCTIONAL_MAP: Dict[Callable, Dict[Tuple[torch.dtype,
                            torch.dtype], Tuple[Callable, Optional[Callable]]]]
CONV_FUNCTIONAL_OPS: Set[Callable]
QBIN_OP_MAPPING: Dict[Union[Callable, str], Callable]
QBIN_RELU_OP_MAPPING: Dict[Union[Callable, str], Callable]


def fold_weight(
    quantized: QuantizedGraphModule, node_name_to_scope: Dict[str, Tuple[str,
                    type]]) -> QuantizedGraphModule: ...


def special_pattern_replacement(model: QuantizedGraphModule): ...
