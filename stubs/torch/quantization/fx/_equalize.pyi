# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fx._equalize import (
    calculate_equalization_scale as calculate_equalization_scale,
)
from torch.ao.quantization.fx._equalize import (
    clear_weight_quant_obs_node as clear_weight_quant_obs_node,
)
from torch.ao.quantization.fx._equalize import convert_eq_obs as convert_eq_obs
from torch.ao.quantization.fx._equalize import (
    default_equalization_qconfig as default_equalization_qconfig,
)
from torch.ao.quantization.fx._equalize import (
    EqualizationQConfig as EqualizationQConfig,
)
from torch.ao.quantization.fx._equalize import (
    fused_module_supports_equalization as fused_module_supports_equalization,
)
from torch.ao.quantization.fx._equalize import (
    get_equalization_qconfig_dict as get_equalization_qconfig_dict,
)
from torch.ao.quantization.fx._equalize import (
    get_layer_sqnr_dict as get_layer_sqnr_dict,
)
from torch.ao.quantization.fx._equalize import (
    get_op_node_and_weight_eq_obs as get_op_node_and_weight_eq_obs,
)
from torch.ao.quantization.fx._equalize import (
    input_equalization_observer as input_equalization_observer,
)
from torch.ao.quantization.fx._equalize import (
    is_equalization_observer as is_equalization_observer,
)
from torch.ao.quantization.fx._equalize import (
    maybe_get_next_equalization_scale as maybe_get_next_equalization_scale,
)
from torch.ao.quantization.fx._equalize import (
    maybe_get_next_input_eq_obs as maybe_get_next_input_eq_obs,
)
from torch.ao.quantization.fx._equalize import (
    maybe_get_weight_eq_obs_node as maybe_get_weight_eq_obs_node,
)
from torch.ao.quantization.fx._equalize import (
    nn_module_supports_equalization as nn_module_supports_equalization,
)
from torch.ao.quantization.fx._equalize import (
    node_supports_equalization as node_supports_equalization,
)
from torch.ao.quantization.fx._equalize import remove_node as remove_node
from torch.ao.quantization.fx._equalize import reshape_scale as reshape_scale
from torch.ao.quantization.fx._equalize import (
    scale_input_observer as scale_input_observer,
)
from torch.ao.quantization.fx._equalize import (
    scale_weight_functional as scale_weight_functional,
)
from torch.ao.quantization.fx._equalize import (
    scale_weight_node as scale_weight_node,
)
from torch.ao.quantization.fx._equalize import (
    update_obs_for_equalization as update_obs_for_equalization,
)
from torch.ao.quantization.fx._equalize import (
    weight_equalization_observer as weight_equalization_observer,
)
