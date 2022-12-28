# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Dict, Tuple

import torch
from _typeshed import Incomplete
from torch.ao.quantization import (
    is_activation_post_process as is_activation_post_process,
)
from torch.fx.node import map_aggregate as map_aggregate

from . import auto_trace_rewriter as auto_trace_rewriter
from .model_utils import (
    attach_op_convert_info_to_model as attach_op_convert_info_to_model,
)
from .model_utils import (
    attach_output_convert_info_to_model as attach_output_convert_info_to_model,
)
from .model_utils import (
    attach_scale_zp_values_to_model as attach_scale_zp_values_to_model,
)
from .model_utils import (
    pack_weights_for_functionals as pack_weights_for_functionals,
)
from .quantization_state import AutoQuantizationState as AutoQuantizationState
from .utils import (
    AutoQuantizationStateModuleDict as AutoQuantizationStateModuleDict,
)
from .utils import (
    get_fqn_valid_for_module_dict_key as get_fqn_valid_for_module_dict_key,
)
from .utils import get_module_hook_type as get_module_hook_type
from .utils import get_torch_function_hook_type as get_torch_function_hook_type
from .utils import HookType as HookType
from .utils import is_leaf as is_leaf
from .utils import OpQuantizeabilityType as OpQuantizeabilityType
from .utils import trace_with_inputs as trace_with_inputs


logger: Incomplete
enable_logging: bool


def add_auto_observation(
    model: torch.nn.Module, qconfig_dict: Dict[str, Any],
    example_inputs: Tuple[Any], input_dtypes: Any = ...,
    prepare_custom_config_dict: Dict[str, Any] = ...) -> torch.nn.Module: ...


def add_auto_convert(module: torch.nn.Module) -> torch.nn.Module: ...
