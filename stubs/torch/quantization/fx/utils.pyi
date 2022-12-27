# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fx.utils import (
    all_node_args_have_no_tensors as all_node_args_have_no_tensors,
)
from torch.ao.quantization.fx.utils import (
    assert_and_get_unique_device as assert_and_get_unique_device,
)
from torch.ao.quantization.fx.utils import (
    create_getattr_from_value as create_getattr_from_value,
)
from torch.ao.quantization.fx.utils import (
    create_qparam_nodes as create_qparam_nodes,
)
from torch.ao.quantization.fx.utils import (
    get_custom_module_class_keys as get_custom_module_class_keys,
)
from torch.ao.quantization.fx.utils import (
    get_linear_prepack_op_for_dtype as get_linear_prepack_op_for_dtype,
)
from torch.ao.quantization.fx.utils import (
    get_new_attr_name_with_prefix as get_new_attr_name_with_prefix,
)
from torch.ao.quantization.fx.utils import (
    get_non_observable_arg_indexes_and_types as get_non_observable_arg_indexes_and_types,
)
from torch.ao.quantization.fx.utils import (
    get_per_tensor_qparams as get_per_tensor_qparams,
)
from torch.ao.quantization.fx.utils import get_qconv_op as get_qconv_op
from torch.ao.quantization.fx.utils import (
    get_qconv_prepack_op as get_qconv_prepack_op,
)
from torch.ao.quantization.fx.utils import (
    graph_module_from_producer_nodes as graph_module_from_producer_nodes,
)
from torch.ao.quantization.fx.utils import graph_pretty_str as graph_pretty_str
from torch.ao.quantization.fx.utils import (
    is_get_tensor_info_node as is_get_tensor_info_node,
)
from torch.ao.quantization.fx.utils import (
    maybe_get_next_module as maybe_get_next_module,
)
from torch.ao.quantization.fx.utils import (
    node_return_type_is_int as node_return_type_is_int,
)
from torch.ao.quantization.fx.utils import quantize_node as quantize_node
