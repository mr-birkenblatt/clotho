# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch.nn.parameter import Parameter as Parameter


class _LearnableFakeQuantize(torch.ao.quantization.FakeQuantizeBase):
    quant_min: Incomplete
    quant_max: Incomplete
    use_grad_scaling: Incomplete
    scale: Incomplete
    zero_point: Incomplete
    activation_post_process: Incomplete
    dtype: Incomplete
    qscheme: Incomplete
    ch_axis: Incomplete
    bitwidth: Incomplete

    def __init__(
        self, observer, quant_min: int = ..., quant_max: int = ...,
        scale: float = ..., zero_point: float = ..., channel_len: int = ...,
        use_grad_scaling: bool = ..., **observer_kwargs) -> None: ...

    def enable_param_learning(self): ...
    def enable_static_estimate(self) -> None: ...
    def enable_static_observation(self) -> None: ...
    def toggle_observer_update(self, enabled: bool = ...): ...
    def enable_observer(self, enabled: bool = ...) -> None: ...
    def toggle_qparam_learning(self, enabled: bool = ...): ...
    def toggle_fake_quant(self, enabled: bool = ...): ...
    def observe_quant_params(self) -> None: ...
    def calculate_qparams(self): ...
    def forward(self, X): ...
