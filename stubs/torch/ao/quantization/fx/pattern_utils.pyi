# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from ..fake_quantize import as, FixedQParamsFakeQuantize


        FixedQParamsFakeQuantize
from typing import Any, Dict, List, Optional, Tuple

from _typeshed import Incomplete
from torch.ao.quantization.quantization_types import Pattern as Pattern
from torch.fx.graph import Node as Node

from ..observer import ObserverBase as ObserverBase
from ..qconfig import QConfigAny as QConfigAny


QuantizeHandler = Any
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
        QConfigAny]
DEFAULT_FUSION_PATTERNS: Incomplete


def register_fusion_pattern(pattern): ...


def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]: ...


DEFAULT_QUANTIZATION_PATTERNS: Incomplete
DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP: Incomplete
DEFAULT_OUTPUT_OBSERVER_MAP: Incomplete


def register_quant_pattern(
    pattern, fixed_qparams_observer: Incomplete | None = ...): ...


def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]: ...


def get_default_output_activation_post_process_map(
    is_training) -> Dict[Pattern, ObserverBase]: ...


def sorted_patterns_dict(
    patterns_dict: Dict[Pattern, QuantizeHandler]) -> Dict[Pattern,
        QuantizeHandler]: ...
