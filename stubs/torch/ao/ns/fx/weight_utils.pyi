# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node

from .ns_types import NSSingleResultType as NSSingleResultType
from .ns_types import NSSingleResultValuesType as NSSingleResultValuesType
from .utils import get_target_type_str as get_target_type_str
from .utils import getattr_from_fqn as getattr_from_fqn
from .utils import (
    return_first_non_observer_node as return_first_non_observer_node,
)


toq: Incomplete


def mod_weight_detach(mod: nn.Module) -> torch.Tensor: ...


def mod_0_weight_detach(mod: nn.Module) -> torch.Tensor: ...


def mod_weight_bias_0(mod: nn.Module) -> torch.Tensor: ...


def get_lstm_weight(mod: nn.Module) -> List[torch.Tensor]: ...


def get_qlstm_weight(mod: nn.Module) -> List[torch.Tensor]: ...


def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor: ...


def get_linear_mod_weight(mod: nn.Module) -> torch.Tensor: ...


def get_lstm_mod_weights(mod: nn.Module) -> List[torch.Tensor]: ...


def get_conv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor: ...


def get_qconv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor: ...


def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor: ...


def get_qlinear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor: ...


def get_op_to_type_to_weight_extraction_fn(
    ) -> Dict[str, Dict[Callable, Callable]]: ...


def extract_weight_from_node(
    node: Node, gm: GraphModule,
    op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable,
                            Callable]]] = ...) -> Optional[
        NSSingleResultType]: ...
