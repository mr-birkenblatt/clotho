from torch.ao.quantization.fuser_method_mappings import (
    DEFAULT_OP_LIST_TO_FUSER_METHOD as DEFAULT_OP_LIST_TO_FUSER_METHOD,
)
from torch.ao.quantization.fuser_method_mappings import (
    fuse_conv_bn as fuse_conv_bn,
)
from torch.ao.quantization.fuser_method_mappings import (
    fuse_conv_bn_relu as fuse_conv_bn_relu,
)
from torch.ao.quantization.fuser_method_mappings import (
    fuse_linear_bn as fuse_linear_bn,
)
from torch.ao.quantization.fuser_method_mappings import (
    get_fuser_method as get_fuser_method,
)
