# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from torch.ao.quantization.observer import ABC as ABC
from torch.ao.quantization.observer import (
    default_debug_observer as default_debug_observer,
)
from torch.ao.quantization.observer import (
    default_dynamic_quant_observer as default_dynamic_quant_observer,
)
from torch.ao.quantization.observer import (
    default_float_qparams_observer as default_float_qparams_observer,
)
from torch.ao.quantization.observer import (
    default_histogram_observer as default_histogram_observer,
)
from torch.ao.quantization.observer import default_observer as default_observer
from torch.ao.quantization.observer import (
    default_per_channel_weight_observer as default_per_channel_weight_observer,
)
from torch.ao.quantization.observer import (
    default_placeholder_observer as default_placeholder_observer,
)
from torch.ao.quantization.observer import (
    default_weight_observer as default_weight_observer,
)
from torch.ao.quantization.observer import (
    get_observer_state_dict as get_observer_state_dict,
)
from torch.ao.quantization.observer import (
    HistogramObserver as HistogramObserver,
)
from torch.ao.quantization.observer import (
    load_observer_state_dict as load_observer_state_dict,
)
from torch.ao.quantization.observer import MinMaxObserver as MinMaxObserver
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver as MovingAverageMinMaxObserver,
)
from torch.ao.quantization.observer import (
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.observer import NoopObserver as NoopObserver
from torch.ao.quantization.observer import ObserverBase as ObserverBase
from torch.ao.quantization.observer import (
    PerChannelMinMaxObserver as PerChannelMinMaxObserver,
)
from torch.ao.quantization.observer import (
    PlaceholderObserver as PlaceholderObserver,
)
from torch.ao.quantization.observer import (
    RecordingObserver as RecordingObserver,
)
