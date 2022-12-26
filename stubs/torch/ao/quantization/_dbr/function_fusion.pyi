from typing import Callable, Dict, Optional, Tuple

from .mappings import (
    known_function_fusion_patterns_and_replacements as known_function_fusion_patterns_and_replacements,
)
from .utils import FusionInfo as FusionInfo
from .utils import (
    get_producer_of_seen_q_op_info as get_producer_of_seen_q_op_info,
)
from .utils import get_users_of_seen_q_op_info as get_users_of_seen_q_op_info
from .utils import SeenQOpInfo as SeenQOpInfo


def pattern_is_match(fusion_pattern: Tuple[Callable, ...], cur_seen_q_op_info: Optional[SeenQOpInfo], idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo]) -> bool: ...
def get_seen_q_op_info_of_start_of_fusion(seen_q_op_info_end_of_fusion: SeenQOpInfo, idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo]) -> SeenQOpInfo: ...
def get_seen_q_op_info_of_end_of_fusion(seen_q_op_info_start_of_fusion: SeenQOpInfo, idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo]) -> SeenQOpInfo: ...
def match_fusion_patterns(idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo]): ...
