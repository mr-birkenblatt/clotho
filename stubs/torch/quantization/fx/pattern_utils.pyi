from torch.ao.quantization.fx.pattern_utils import (
    get_default_fusion_patterns as get_default_fusion_patterns,
)
from torch.ao.quantization.fx.pattern_utils import (
    get_default_output_activation_post_process_map as get_default_output_activation_post_process_map,
)
from torch.ao.quantization.fx.pattern_utils import (
    get_default_quant_patterns as get_default_quant_patterns,
)
from torch.ao.quantization.fx.pattern_utils import MatchResult as MatchResult
from torch.ao.quantization.fx.pattern_utils import (
    QuantizeHandler as QuantizeHandler,
)
from torch.ao.quantization.fx.pattern_utils import (
    register_fusion_pattern as register_fusion_pattern,
)
from torch.ao.quantization.fx.pattern_utils import (
    register_quant_pattern as register_quant_pattern,
)
