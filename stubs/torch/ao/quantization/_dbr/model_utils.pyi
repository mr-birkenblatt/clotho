# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch

from .mappings import conv_ops as conv_ops


        conv_prepack_fns as conv_prepack_fns
from _typeshed import Incomplete
from torch.quantization import FakeQuantizeBase as FakeQuantizeBase

from .quantization_state import AutoQuantizationState as AutoQuantizationState


        ObserverBase as ObserverBase

toq: Incomplete


def pack_weights_for_functionals(module: torch.nn.Module) -> None: ...


def attach_scale_zp_values_to_model(module: torch.nn.Module) -> None: ...


def attach_op_convert_info_to_model(module: torch.nn.Module) -> None: ...


def attach_output_convert_info_to_model(module: torch.nn.Module) -> None: ...
