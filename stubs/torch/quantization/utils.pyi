# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.utils import activation_dtype as activation_dtype


        activation_is_int8_quantized as activation_is_int8_quantized,
        activation_is_statically_quantized as
        activation_is_statically_quantized,
        calculate_qmin_qmax as calculate_qmin_qmax,
        check_min_max_valid as check_min_max_valid,
        get_combined_dict as get_combined_dict,
        get_qconfig_dtypes as get_qconfig_dtypes,
        get_qparam_dict as get_qparam_dict, get_quant_type as get_quant_type,
        get_swapped_custom_module_class as get_swapped_custom_module_class,
        getattr_from_fqn as getattr_from_fqn,
        is_per_channel as is_per_channel, is_per_tensor as is_per_tensor,
        weight_dtype as weight_dtype,
        weight_is_quantized as weight_is_quantized,
        weight_is_statically_quantized as weight_is_statically_quantized
