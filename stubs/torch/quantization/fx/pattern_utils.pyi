# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fx.pattern_utils import MatchResult as MatchResult
from torch.ao.quantization.fx.pattern_utils import (
    QuantizeHandler as QuantizeHandler,
)


        get_default_fusion_patterns as get_default_fusion_patterns,
        get_default_output_activation_post_process_map as \
        get_default_output_activation_post_process_map,
        get_default_quant_patterns as get_default_quant_patterns,
        register_fusion_pattern as register_fusion_pattern,
        register_quant_pattern as register_quant_pattern
