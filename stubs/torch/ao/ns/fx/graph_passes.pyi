# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .ns_types import NSNodeTargetType as NSNodeTargetType


        NSSingleResultValuesType as NSSingleResultValuesType,
        NSSubgraph as NSSubgraph
from .utils import NodeInputOrOutputType as NodeInputOrOutputType


        get_arg_indices_of_inputs_to_log as get_arg_indices_of_inputs_to_log,
        get_node_first_input_and_output_type as \
        get_node_first_input_and_output_type,
        get_node_input_qparams as get_node_input_qparams,
        get_number_of_non_param_args as get_number_of_non_param_args,
        get_target_type_str as get_target_type_str,
        getattr_from_fqn as getattr_from_fqn,
        op_type_supports_shadowing as op_type_supports_shadowing,
        return_first_non_observer_node as return_first_non_observer_node
from typing import Callable, Dict, Optional, Set, Tuple

from torch.ao.ns.fx.mappings import (
    get_node_type_to_io_type_map as get_node_type_to_io_type_map,
)
from torch.ao.quantization.fx.utils import (
    get_new_attr_name_with_prefix as get_new_attr_name_with_prefix,
)
from torch.ao.quantization.quantize import (
    is_activation_post_process as is_activation_post_process,
)
from torch.fx import GraphModule as GraphModule
from torch.fx import map_arg as map_arg
from torch.fx.graph import Graph as Graph
from torch.fx.graph import Node as Node


def add_loggers_to_model(
    gm: GraphModule, node_to_instrument_inputs_to_ref_node_name: Dict[Node,
    Tuple[str, str]], node_to_instrument_outputs_to_ref_node_name: Dict[Node,
    Tuple[str, str]], logger_cls: Callable,
    model_name: str) -> GraphModule: ...


def create_a_shadows_b(
    name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule,
    matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]],
    logger_cls: Callable, should_log_inputs: bool,
    node_type_to_io_type_map: Optional[Dict[str,
    Set[NSNodeTargetType]]] = ...) -> GraphModule: ...
