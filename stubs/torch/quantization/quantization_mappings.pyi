# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.quantization_mappings import (
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS as DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
)


        DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS as \
        DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS,
        DEFAULT_MODULE_TO_ACT_POST_PROCESS as \
        DEFAULT_MODULE_TO_ACT_POST_PROCESS,
        DEFAULT_QAT_MODULE_MAPPINGS as DEFAULT_QAT_MODULE_MAPPINGS,
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS as \
        DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
        DEFAULT_STATIC_QUANT_MODULE_MAPPINGS as \
        DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,
        get_default_compare_output_module_list as \
        get_default_compare_output_module_list,
        get_default_dynamic_quant_module_mappings as \
        get_default_dynamic_quant_module_mappings,
        get_default_float_to_quantized_operator_mappings as \
        get_default_float_to_quantized_operator_mappings,
        get_default_qat_module_mappings as get_default_qat_module_mappings,
        get_default_qconfig_propagation_list as \
        get_default_qconfig_propagation_list,
        get_default_static_quant_module_mappings as \
        get_default_static_quant_module_mappings,
        get_dynamic_quant_module_class as get_dynamic_quant_module_class,
        get_quantized_operator as get_quantized_operator,
        get_static_quant_module_class as get_static_quant_module_class,
        no_observer_set as no_observer_set
