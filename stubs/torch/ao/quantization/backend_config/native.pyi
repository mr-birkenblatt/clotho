from typing import NamedTuple

from _typeshed import Incomplete


class _ConvMetadata(NamedTuple):
    root: Incomplete
    transpose: Incomplete
    bn: Incomplete
    reference: Incomplete
    transpose_reference: Incomplete
    fused_conv_relu: Incomplete
    fused_conv_bn: Incomplete
    fused_conv_bn_relu: Incomplete
    qat: Incomplete
    relu_qat: Incomplete
    bn_qat: Incomplete
    bn_relu_qat: Incomplete
    func: Incomplete

def get_native_backend_config_dict(): ...
