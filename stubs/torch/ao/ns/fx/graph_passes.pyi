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

from .ns_types import NSNodeTargetType as NSNodeTargetType
from .ns_types import NSSingleResultValuesType as NSSingleResultValuesType
from .ns_types import NSSubgraph as NSSubgraph
from .utils import (
    get_arg_indices_of_inputs_to_log as get_arg_indices_of_inputs_to_log,
)
from .utils import (
    get_node_first_input_and_output_type as get_node_first_input_and_output_type,
)
from .utils import get_node_input_qparams as get_node_input_qparams
from .utils import get_number_of_non_param_args as get_number_of_non_param_args
from .utils import get_target_type_str as get_target_type_str
from .utils import getattr_from_fqn as getattr_from_fqn
from .utils import NodeInputOrOutputType as NodeInputOrOutputType
from .utils import op_type_supports_shadowing as op_type_supports_shadowing
from .utils import (
    return_first_non_observer_node as return_first_non_observer_node,
)


def add_loggers_to_model(gm: GraphModule, node_to_instrument_inputs_to_ref_node_name: Dict[Node, Tuple[str, str]], node_to_instrument_outputs_to_ref_node_name: Dict[Node, Tuple[str, str]], logger_cls: Callable, model_name: str) -> GraphModule: ...
def create_a_shadows_b(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]], logger_cls: Callable, should_log_inputs: bool, node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]] = ...) -> GraphModule: ...
