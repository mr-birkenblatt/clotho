# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.qconfig import QConfig as QConfig


        QConfigAny as QConfigAny, QConfigDynamic as QConfigDynamic,
        add_module_to_qconfig_obs_ctr as add_module_to_qconfig_obs_ctr,
        assert_valid_qconfig as assert_valid_qconfig,
        default_activation_only_qconfig as default_activation_only_qconfig,
        default_debug_qconfig as default_debug_qconfig,
        default_dynamic_qconfig as default_dynamic_qconfig,
        default_per_channel_qconfig as default_per_channel_qconfig,
        default_qat_qconfig as default_qat_qconfig,
        default_qat_qconfig_v2 as default_qat_qconfig_v2,
        default_qconfig as default_qconfig,
        default_weight_only_qconfig as default_weight_only_qconfig,
        float16_dynamic_qconfig as float16_dynamic_qconfig,
        float16_static_qconfig as float16_static_qconfig,
        float_qparams_weight_only_qconfig as
        float_qparams_weight_only_qconfig,
        get_default_qat_qconfig as get_default_qat_qconfig,
        get_default_qconfig as get_default_qconfig,
        per_channel_dynamic_qconfig as per_channel_dynamic_qconfig,
        qconfig_equals as qconfig_equals
