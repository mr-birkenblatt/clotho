from typing import Dict, Optional, Set

from _typeshed import Incomplete
from torch.ao.quantization.backend_config import (
    get_native_backend_config_dict as get_native_backend_config_dict,
)

from .ns_types import NSNodeTargetType as NSNodeTargetType


toq: Incomplete

def get_base_name_to_sets_of_related_ops() -> Dict[str, Set[NSNodeTargetType]]: ...
def get_base_name_for_op(base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]], op: NSNodeTargetType) -> Optional[str]: ...
def add_op_to_sets_of_related_ops(base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]], op: NSNodeTargetType, related_op: Optional[NSNodeTargetType]) -> None: ...
def get_node_type_to_io_type_map() -> Dict[str, Set[NSNodeTargetType]]: ...
def get_unmatchable_types_map() -> Dict[str, Set[NSNodeTargetType]]: ...
