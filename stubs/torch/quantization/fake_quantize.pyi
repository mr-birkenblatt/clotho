# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fake_quantize import FakeQuantize as FakeQuantize


        FakeQuantizeBase as FakeQuantizeBase,
        FixedQParamsFakeQuantize as FixedQParamsFakeQuantize,
        FusedMovingAvgObsFakeQuantize as FusedMovingAvgObsFakeQuantize,
        default_fake_quant as default_fake_quant,
        default_fixed_qparams_range_0to1_fake_quant as
        default_fixed_qparams_range_0to1_fake_quant,
        default_fixed_qparams_range_neg1to1_fake_quant as
        default_fixed_qparams_range_neg1to1_fake_quant,
        default_fused_act_fake_quant as default_fused_act_fake_quant,
        default_fused_per_channel_wt_fake_quant as
        default_fused_per_channel_wt_fake_quant,
        default_fused_wt_fake_quant as default_fused_wt_fake_quant,
        default_histogram_fake_quant as default_histogram_fake_quant,
        default_per_channel_weight_fake_quant as
        default_per_channel_weight_fake_quant,
        default_weight_fake_quant as default_weight_fake_quant,
        disable_fake_quant as disable_fake_quant,
        disable_observer as disable_observer,
        enable_fake_quant as enable_fake_quant,
        enable_observer as enable_observer
