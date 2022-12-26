from typing import Any, Dict, List, Optional, Tuple

from _typeshed import Incomplete
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.fx.graph import Node as Node

from ..fake_quantize import (
    FixedQParamsFakeQuantize as FixedQParamsFakeQuantize,
)
from ..observer import ObserverBase as ObserverBase
from ..qconfig import QConfigAny as QConfigAny


QuantizeHandler = Any
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler, QConfigAny]
DEFAULT_FUSION_PATTERNS: Incomplete

def register_fusion_pattern(pattern): ...
def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]: ...

DEFAULT_QUANTIZATION_PATTERNS: Incomplete
DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP: Incomplete
DEFAULT_OUTPUT_OBSERVER_MAP: Incomplete

def register_quant_pattern(pattern, fixed_qparams_observer: Incomplete | None = ...): ...
def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]: ...
def get_default_output_activation_post_process_map(is_training) -> Dict[Pattern, ObserverBase]: ...
def sorted_patterns_dict(patterns_dict: Dict[Pattern, QuantizeHandler]) -> Dict[Pattern, QuantizeHandler]: ...