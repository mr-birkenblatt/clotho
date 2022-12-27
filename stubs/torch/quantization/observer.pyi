# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.ao.quantization.observer import ABC as ABC


        HistogramObserver as HistogramObserver,
        MinMaxObserver as MinMaxObserver,
        MovingAverageMinMaxObserver as MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver as
        MovingAveragePerChannelMinMaxObserver, NoopObserver as NoopObserver,
        ObserverBase as ObserverBase,
        PerChannelMinMaxObserver as PerChannelMinMaxObserver,
        PlaceholderObserver as PlaceholderObserver,
        RecordingObserver as RecordingObserver,
        default_debug_observer as default_debug_observer,
        default_dynamic_quant_observer as default_dynamic_quant_observer,
        default_float_qparams_observer as default_float_qparams_observer,
        default_histogram_observer as default_histogram_observer,
        default_observer as default_observer,
        default_per_channel_weight_observer as
        default_per_channel_weight_observer,
        default_placeholder_observer as default_placeholder_observer,
        default_weight_observer as default_weight_observer,
        get_observer_state_dict as get_observer_state_dict,
        load_observer_state_dict as load_observer_state_dict
