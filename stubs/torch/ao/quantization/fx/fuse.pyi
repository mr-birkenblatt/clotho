# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Dict, Optional

from torch.ao.quantization.quantization_types import NodePattern as NodePattern
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.fx import GraphModule as GraphModule
from torch.fx import map_arg as map_arg
from torch.fx import Node as Node
from torch.fx.graph import Graph as Graph

from ..backend_config import (
    get_native_backend_config_dict as get_native_backend_config_dict,
)
from ..backend_config.utils import (
    get_fuser_method_mapping as get_fuser_method_mapping,
)
from ..backend_config.utils import (
    get_fusion_pattern_to_extra_inputs_getter as get_fusion_pattern_to_extra_inputs_getter,
)
from ..backend_config.utils import (
    get_fusion_pattern_to_root_node_getter as get_fusion_pattern_to_root_node_getter,
)
from .backend_config_utils import (
    get_fusion_pattern_to_fuse_handler_cls as get_fusion_pattern_to_fuse_handler_cls,
)
from .fusion_patterns import *
from .graph_module import FusedGraphModule as FusedGraphModule
from .match_utils import is_match as is_match
from .match_utils import MatchAllNode as MatchAllNode
from .pattern_utils import sorted_patterns_dict as sorted_patterns_dict


def fuse(
    model: GraphModule, is_qat: bool,
        fuse_custom_config_dict: Optional[Dict[str, Any]] = ...,
        backend_config_dict: Optional[Dict[str,
                Any]] = ...) -> GraphModule: ...
