from typing import List

import torch

from .function_fusion import pattern_is_match as pattern_is_match
from .mappings import (
    known_module_fusion_patterns as known_module_fusion_patterns,
)
from .utils import get_users_of_seen_q_op_info as get_users_of_seen_q_op_info


def get_module_fusion_fqns(module: torch.nn.Module) -> List[List[str]]: ...
