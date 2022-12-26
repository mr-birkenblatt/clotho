import torch
from _typeshed import Incomplete
from torch.quantization import FakeQuantizeBase as FakeQuantizeBase
from torch.quantization import ObserverBase as ObserverBase

from .mappings import conv_ops as conv_ops
from .mappings import conv_prepack_fns as conv_prepack_fns
from .quantization_state import AutoQuantizationState as AutoQuantizationState


toq: Incomplete

def pack_weights_for_functionals(module: torch.nn.Module) -> None: ...
def attach_scale_zp_values_to_model(module: torch.nn.Module) -> None: ...
def attach_op_convert_info_to_model(module: torch.nn.Module) -> None: ...
def attach_output_convert_info_to_model(module: torch.nn.Module) -> None: ...
