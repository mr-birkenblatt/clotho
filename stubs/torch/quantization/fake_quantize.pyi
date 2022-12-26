from torch.ao.quantization.fake_quantize import (
    default_fake_quant as default_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant as default_fixed_qparams_range_0to1_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_neg1to1_fake_quant as default_fixed_qparams_range_neg1to1_fake_quant,
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
    default_histogram_fake_quant as default_histogram_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_per_channel_weight_fake_quant as default_per_channel_weight_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    default_weight_fake_quant as default_weight_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    disable_fake_quant as disable_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    disable_observer as disable_observer,
)
from torch.ao.quantization.fake_quantize import (
    enable_fake_quant as enable_fake_quant,
)
from torch.ao.quantization.fake_quantize import (
    enable_observer as enable_observer,
)
from torch.ao.quantization.fake_quantize import FakeQuantize as FakeQuantize
from torch.ao.quantization.fake_quantize import (
    FakeQuantizeBase as FakeQuantizeBase,
)
from torch.ao.quantization.fake_quantize import (
    FixedQParamsFakeQuantize as FixedQParamsFakeQuantize,
)
from torch.ao.quantization.fake_quantize import (
    FusedMovingAvgObsFakeQuantize as FusedMovingAvgObsFakeQuantize,
)
