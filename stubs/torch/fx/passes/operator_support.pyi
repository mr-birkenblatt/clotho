import abc
import typing as t

import torch.fx
from _typeshed import Incomplete
from torch.fx._compatibility import compatibility as compatibility

from .shape_prop import TensorMetadata as TensorMetadata
from .tools_common import CALLABLE_NODE_OPS as CALLABLE_NODE_OPS
from .tools_common import get_node_target as get_node_target


TargetTypeName = str
SupportedArgumentDTypes: Incomplete
SupportDict: Incomplete

class OperatorSupportBase(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool: ...

class OperatorSupport(OperatorSupportBase):
    def __init__(self, support_dict: t.Optional[SupportDict] = ...) -> None: ...
    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool: ...

IsNodeSupported: Incomplete

def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase: ...
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase: ...

class OpSupports:
    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase: ...
    @classmethod
    def decline_if_node_in_names(cls, disallow_set: t.Set[str]) -> OperatorSupportBase: ...