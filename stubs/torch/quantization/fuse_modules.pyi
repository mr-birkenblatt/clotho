from torch.ao.quantization.fuse_modules import (
    fuse_known_modules as fuse_known_modules,
)
from torch.ao.quantization.fuse_modules import fuse_modules as fuse_modules
from torch.ao.quantization.fuse_modules import (
    get_fuser_method as get_fuser_method,
)
from torch.quantization.fuser_method_mappings import (
    fuse_conv_bn as fuse_conv_bn,
)
from torch.quantization.fuser_method_mappings import (
    fuse_conv_bn_relu as fuse_conv_bn_relu,
)
