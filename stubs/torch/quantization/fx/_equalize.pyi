# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.fx._equalize import as, EqualizationQConfig


        EqualizationQConfig,
        calculate_equalization_scale as calculate_equalization_scale,
        clear_weight_quant_obs_node as clear_weight_quant_obs_node,
        convert_eq_obs as convert_eq_obs,
        default_equalization_qconfig as default_equalization_qconfig,
        fused_module_supports_equalization as
        fused_module_supports_equalization,
        get_equalization_qconfig_dict as get_equalization_qconfig_dict,
        get_layer_sqnr_dict as get_layer_sqnr_dict,
        get_op_node_and_weight_eq_obs as get_op_node_and_weight_eq_obs,
        input_equalization_observer as input_equalization_observer,
        is_equalization_observer as is_equalization_observer,
        maybe_get_next_equalization_scale as
        maybe_get_next_equalization_scale,
        maybe_get_next_input_eq_obs as maybe_get_next_input_eq_obs,
        maybe_get_weight_eq_obs_node as maybe_get_weight_eq_obs_node,
        nn_module_supports_equalization as nn_module_supports_equalization,
        node_supports_equalization as node_supports_equalization,
        remove_node as remove_node, reshape_scale as reshape_scale,
        scale_input_observer as scale_input_observer,
        scale_weight_functional as scale_weight_functional,
        scale_weight_node as scale_weight_node,
        update_obs_for_equalization as update_obs_for_equalization,
        weight_equalization_observer as weight_equalization_observer
