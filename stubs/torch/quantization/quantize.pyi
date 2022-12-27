# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.quantize import add_observer_ as add_observer_


        add_quant_dequant as add_quant_dequant, convert as convert,
        get_observer_dict as get_observer_dict,
        get_unique_devices_ as get_unique_devices_,
        is_activation_post_process as is_activation_post_process,
        prepare as prepare, prepare_qat as prepare_qat,
        propagate_qconfig_ as propagate_qconfig_, quantize as quantize,
        quantize_dynamic as quantize_dynamic, quantize_qat as quantize_qat,
        register_activation_post_process_hook as
        register_activation_post_process_hook, swap_module as swap_module
