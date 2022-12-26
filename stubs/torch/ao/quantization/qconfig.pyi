from typing import Any, Optional

import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch.ao.quantization.fake_quantize import (
    default_dynamic_fake_quant as default_dynamic_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_embedding_fake_quant as default_embedding_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_embedding_fake_quant_4bit as default_embedding_fake_quant_4bit,
)
from torch.ao.quantization.fake_quantize import (
    default_fake_quant as default_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_fused_act_fake_quant as default_fused_act_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_fused_per_channel_wt_fake_quant as default_fused_per_channel_wt_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_fused_wt_fake_quant as default_fused_wt_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_per_channel_weight_fake_quant as default_per_channel_weight_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_weight_fake_quant as default_weight_fake_quant,
)
from torch.ao.quantization.fake_quantize import FakeQuantize as FakeQuantize
from torch.ao.quantization.fake_quantize import (
    FakeQuantizeBase as FakeQuantizeBase,
)
from torch.ao.quantization.fake_quantize import (
    fused_per_channel_wt_fake_quant_range_neg_127_to_127 as fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)
from torch.ao.quantization.fake_quantize import (
    fused_wt_fake_quant_range_neg_127_to_127 as fused_wt_fake_quant_range_neg_127_to_127,
)
from torch.ao.quantization.fake_quantize import (
    FusedMovingAvgObsFakeQuantize as FusedMovingAvgObsFakeQuantize,
)

from .observer import default_debug_observer as default_debug_observer
from .observer import (
    default_dynamic_quant_observer as default_dynamic_quant_observer,
)
from .observer import (
    default_float_qparams_observer as default_float_qparams_observer,
)
from .observer import (
    default_float_qparams_observer_4bit as default_float_qparams_observer_4bit,
)
from .observer import default_observer as default_observer
from .observer import (
    default_per_channel_weight_observer as default_per_channel_weight_observer,
)
from .observer import (
    default_placeholder_observer as default_placeholder_observer,
)
from .observer import (
    default_reuse_input_observer as default_reuse_input_observer,
)
from .observer import default_weight_observer as default_weight_observer
from .observer import HistogramObserver as HistogramObserver
from .observer import (
    MovingAverageMinMaxObserver as MovingAverageMinMaxObserver,
)
from .observer import NoopObserver as NoopObserver
from .observer import (
    per_channel_weight_observer_range_neg_127_to_127 as per_channel_weight_observer_range_neg_127_to_127,
)
from .observer import PlaceholderObserver as PlaceholderObserver
from .observer import ReuseInputObserver as ReuseInputObserver
from .observer import (
    weight_observer_range_neg_127_to_127 as weight_observer_range_neg_127_to_127,
)


class QConfig:
    def __new__(cls, activation, weight): ...

class QConfigDynamic:
    def __new__(cls, activation=..., weight=...): ...

default_qconfig: Incomplete
default_debug_qconfig: Incomplete
default_per_channel_qconfig: Incomplete
default_dynamic_qconfig: Incomplete
float16_dynamic_qconfig: Incomplete
float16_static_qconfig: Incomplete
per_channel_dynamic_qconfig: Incomplete
float_qparams_weight_only_qconfig: Incomplete
float_qparams_weight_only_qconfig_4bit: Incomplete
default_qat_qconfig: Incomplete
default_dynamic_qat_qconfig: Incomplete
default_weight_only_qconfig: Incomplete
default_activation_only_qconfig: Incomplete
default_qat_qconfig_v2: Incomplete
default_reuse_input_qconfig: Incomplete

def get_default_qconfig(backend: str = ..., version: int = ...): ...

default_symmetric_qnnpack_qconfig: Incomplete
default_per_channel_symmetric_qnnpack_qconfig: Incomplete
default_embedding_qat_qconfig: Incomplete
default_embedding_qat_qconfig_4bit: Incomplete

def get_default_qat_qconfig(backend: str = ..., version: int = ...): ...

default_symmetric_qnnpack_qat_qconfig: Incomplete
default_per_channel_symmetric_qnnpack_qat_qconfig: Incomplete

def get_default_qconfig_dict(backend: str = ..., version: int = ...): ...
def get_default_qat_qconfig_dict(backend: str = ..., version: int = ...): ...
def assert_valid_qconfig(qconfig: Optional[QConfig], mod: torch.nn.Module) -> None: ...
QConfigAny = Optional[QConfig]

def add_module_to_qconfig_obs_ctr(qconfig: QConfigAny, module: Optional[nn.Module]) -> Any: ...
def qconfig_equals(q1: QConfigAny, q2: QConfigAny): ...
def activation_is_memoryless(qconfig: QConfig): ...
def is_reuse_input_qconfig(qconfig: Optional[QConfig]): ...
