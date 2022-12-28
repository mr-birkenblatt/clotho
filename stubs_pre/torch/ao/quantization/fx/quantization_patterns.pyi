# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from abc import ABC
from typing import Any, Callable, Dict, Optional

import torch
from _typeshed import Incomplete
from torch.ao.quantization.quantization_types import NodePattern as NodePattern
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.fx.graph import Node as Node

from .utils import (
    all_node_args_have_no_tensors as all_node_args_have_no_tensors,
)


class QuantizeHandler(ABC):
    node_pattern: Incomplete
    modules: Incomplete
    root_node: Incomplete
    is_custom_module_: Incomplete
    is_standalone_module_: Incomplete
    num_tensor_args: int

    def __init__(
        self, node_pattern: NodePattern, modules: Dict[str, torch.nn.Module],
        root_node_getter: Callable = ..., is_custom_module: bool = ...,
        is_standalone_module: bool = ...) -> None: ...

    def input_output_observed(self) -> bool: ...
    def is_general_tensor_value_op(self) -> bool: ...

    def get_activation_ctr(
        self, qconfig: Any, pattern: Pattern, is_training: bool) -> Optional[
            Callable]: ...

    def is_custom_module(self): ...
    def is_standalone_module(self): ...


class BinaryOpQuantizeHandler(QuantizeHandler):
    ...


class CatQuantizeHandler(QuantizeHandler):
    ...


class ConvReluQuantizeHandler(QuantizeHandler):
    ...


class LinearReLUQuantizeHandler(QuantizeHandler):
    ...


class BatchNormQuantizeHandler(QuantizeHandler):
    ...


class EmbeddingQuantizeHandler(QuantizeHandler):
    ...


class RNNDynamicQuantizeHandler(QuantizeHandler):
    ...


class DefaultNodeQuantizeHandler(QuantizeHandler):
    ...


class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    ...


class CopyNodeQuantizeHandler(QuantizeHandler):
    ...


class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler):
    ...


class CustomModuleQuantizeHandler(QuantizeHandler):
    ...


class StandaloneModuleQuantizeHandler(QuantizeHandler):
    ...
